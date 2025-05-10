import torch, os
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, GINConv, SAGEConv, GATConv, GCN, global_mean_pool
from typing import List
from utils import *
import ipdb


class Discriminator(nn.Module):
    def __init__(
            self,
            in_channels, 
            hidden_channels, 
            out_channels, 
            num_layers = 2,
            dropout = 0.0,
            *args, 
            **kwargs
        ) -> None:
        super(Discriminator, self).__init__(*args, **kwargs)
        self.linear1 = nn.Linear(in_channels, hidden_channels)
        self.linear2 = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, graph_embed):
        x = F.relu(self.linear1(graph_embed))
        x = F.dropout(x, p=self.dropout, training=self.training)
        out = self.linear2(x)
        return out


class PretrainedModel(nn.Module):
    def __init__(
            self, gnn_type,
            in_channels, 
            hidden_channels, 
            out_channels, 
            num_layers = 2,
            dropout = 0.0, 
            with_bn = False,
            with_head = True,
            *args, 
            **kwargs
        ) -> None:
        super(PretrainedModel, self).__init__(*args, **kwargs)
        self.gnn_type = gnn_type
        self.num_layers = num_layers
        self.dropout = dropout
        self.with_bn = with_bn
        self.with_head = with_head
        self.gnn_layers = nn.ModuleList([self.get_gnn_layer(gnn_type, in_channels, hidden_channels)])
        if with_bn:
            self.bns = nn.ModuleList([nn.BatchNorm1d(hidden_channels)])
        num_hid_layer = num_layers - 1 if with_head else num_layers - 2
        for _ in range(num_hid_layer):
            self.gnn_layers.append(self.get_gnn_layer(gnn_type, hidden_channels, hidden_channels))
            if with_bn:
                self.bns.append(nn.BatchNorm1d(hidden_channels))
        if with_head:
            self.linear_decoder = nn.Linear(hidden_channels, out_channels)
        else:
            self.gnn_decoder = self.get_gnn_layer(gnn_type, hidden_channels, out_channels)
    
    def get_gnn_layer(self, gnn_type, in_channels, out_channels):
        if gnn_type == "gcn":
            return GCNConv(in_channels, out_channels)
        elif gnn_type == "gat":
            return GATConv(in_channels, out_channels)
        elif gnn_type == "gin":
            return GINConv(nn.Sequential(nn.Linear(in_channels, out_channels)))
        elif gnn_type == "sage":
            return SAGEConv(in_channels, out_channels)
        else:
            raise Exception("The model is not implemented!")
    
    def forward(self, graph_batch, decoder = True, device = None, prompt_params=None):
        if isinstance(graph_batch, list):
            graph_batch = Batch.from_data_list(graph_batch)
        if device is not None:
            graph_batch = graph_batch.to(device)
        x_p = []
        # ipdb.set_trace()
        for j in range(self.num_layers+1):
            x = graph_batch.x.clone()
            for i, layer in enumerate(self.gnn_layers):
                if (i == j) and (prompt_params is not None):
                    x *= prompt_params[j]
                x = layer(x, graph_batch.edge_index)
                x = self.bns[i](x) if self.with_bn else x
                x = F.relu(x)
                if i < len(self.gnn_layers) - 1:
                    x = F.dropout(x, p=self.dropout, training=self.training)
            if not decoder:
                if prompt_params is None:
                    return "scores", x
                x_p.append(x)
            if prompt_params is not None:
                continue
            else:
                if not self.with_head:
                    x = F.dropout(x, p=self.dropout, training=self.training)
                    x = self.gnn_decoder(x, graph_batch.edge_index)
                    scores = global_mean_pool(x, graph_batch.batch)
                if self.with_head:
                    x = global_mean_pool(x, graph_batch.batch)
                    scores = F.dropout(x, p=self.dropout, training=self.training)
                    scores = self.linear_decoder(scores)
                return scores, x
        return "scores", x_p
    

class BasePrompt(nn.Module):
    def __init__(
            self,
            emb_dim,
            h_dim,
            output_dim,
            prompt_fn = "gpf_plus",
            token_num = 30,
            cross_prune=0.1, 
            inner_prune=0.3,
            attn_dropout=0.3,
            input_dropout=0.3,
            attn_with_param=False,
        ) -> None:
        super(BasePrompt, self).__init__()
        self.prefix_prompt = True
        self.head = nn.Linear(h_dim, output_dim)
        self.emb_dim = emb_dim
        self.prompt_fn = prompt_fn
        self.token_embeds = torch.nn.Parameter(torch.empty(token_num, emb_dim))
        torch.nn.init.kaiming_uniform_(self.token_embeds, nonlinearity='leaky_relu', mode='fan_in', a=0.01)
        if prompt_fn == "gpf_plus":
            self.attn_with_param = attn_with_param
            if attn_with_param:
                self.attn_linear = nn.Linear(2*emb_dim, 1)
            self.attn_dropout = attn_dropout
            self.prompt = self.gpf_plus
        elif prompt_fn == "add_tokens":
            self.inner_prune = inner_prune
            self.cross_prune = cross_prune
            self.pg = self.pg_construct()
            self.prompt = self.add_token
        elif prompt_fn == "tucker":
            self.core = nn.Parameter(torch.Tensor(emb_dim, emb_dim, emb_dim))
            nn.init.xavier_uniform_(self.core, gain=nn.init.calculate_gain('relu'))
            self.attn_dropout = attn_dropout
            self.input_dropout = input_dropout
            self.prompt = self.tucker
        else:
            raise Exception("The prompting function is not implemented")

    @property
    def name(self,):
        return "ugprompt"

    def get_inner_edges(self, x):
        token_sim = x @ x.T
        token_sim.fill_diagonal_(.0)
        token_sim = torch.sigmoid(token_sim)
        inner_adj = torch.where(token_sim < self.inner_prune, 0, token_sim)
        edge_index = inner_adj.nonzero().T.contiguous()
        return edge_index

    def pg_construct(self,):
        token_sim = self.token_embeds @ self.token_embeds.T
        token_sim.fill_diagonal_(.0)
        token_sim = torch.sigmoid(token_sim)
        inner_adj = torch.where(token_sim < self.inner_prune, 0, token_sim)
        edge_index = inner_adj.nonzero().t().contiguous()
        pg = Data(x=self.token_embeds, edge_index=edge_index, y=torch.tensor([0]).long())
        return pg

    def simmatToadj(self, adjacency_matrix):
        adjacency_matrix = adjacency_matrix.triu(diagonal=1)
        adj_list = torch.where(adjacency_matrix >= 0.5)
        edge_weights = adjacency_matrix[adj_list]
        adj_list = torch.cat(
            (adj_list[0][None, :], adj_list[1][None, :]),
            dim=0)
        return adj_list, edge_weights
    
    def add_token(self, graphs):
        self.pg.to(graphs[0].x.device)
        for graph in graphs:
            cross_dot = self.pg.x @ graph.x.T
            cross_sim = torch.sigmoid(cross_dot)  # 0-1 from prompt to input graph
            cross_adj = torch.where(cross_sim < self.cross_prune, 0, cross_sim)
            cross_edge_index = cross_adj.nonzero().T.contiguous()
            added_tokens = cross_edge_index[0].unique()
            x = torch.cat([self.pg.x[added_tokens], graph.x], dim=0)
            g_edge_index = graph.edge_index + added_tokens.size(0)
            cross_edge_index[0] = (added_tokens[None, :] == cross_edge_index[0][:, None]).nonzero()[:, 1]
            cross_edge_index[1] = cross_edge_index[1] + added_tokens.size(0)
            inner_edge_index = self.get_inner_edges(self.pg.x[added_tokens])
            edge_index = torch.cat([inner_edge_index, g_edge_index, cross_edge_index], dim=1)
            graph.edge_index = edge_index
            graph.x = x
        return graphs

    def gpf_plus(self, graphs):
        for graph in graphs:
            if self.attn_with_param:
                n_t = self.token_embeds.size(0)
                n_x = graph.x.size(0)
                x = torch.cat((
                    graph.x[:, None, :].tile(1, n_t, 1),
                    self.token_embeds[None, :, :].tile(n_x, 1, 1)
                ), dim=-1)
                x = self.attn_linear(x).squeeze()
                attn_scores = F.softmax(x, dim = 1)
            else:
                attn_scores = F.softmax(graph.x @ self.token_embeds.T, dim = 1)
            attn_scores = F.dropout(attn_scores, self.attn_dropout, training=self.training)
            prompt = attn_scores @ self.token_embeds
            graph.x = graph.x + prompt
        return graphs
    
    def tucker(self, graphs):
        core = self.core.view(self.emb_dim, -1)
        for graph in graphs:
            attn_scores = F.softmax(graph.x @ self.token_embeds.T, dim = 1)
            attn_scores = F.dropout(attn_scores, self.attn_dropout, training=self.training)
            xr = attn_scores @ self.token_embeds
            xr = torch.mm(xr, core)
            xr = xr.view(-1, self.emb_dim, self.emb_dim)
            x = graph.x.view(-1, 1, self.emb_dim).contiguous()
            x = F.dropout(x, self.input_dropout, training=self.training)
            x = torch.bmm(x, xr)
            # TODO: We can also addition instead of substitution s.t. graph.x = graph.x + x.view(-1, self.emb_dim) similar to gpf_plus
            graph.x = x.view(-1, self.emb_dim)
        return graphs

    def forward(self, graphs, device = None):
        if isinstance(graphs, Batch):
            graphs = graphs.to_data_list()
        if device is not None:
            graphs = [graph.to(device) for graph in graphs]
        graphs = self.prompt(graphs)
        if not isinstance(graphs, Batch):
            assert isinstance(graphs, List)
            graphs = Batch.from_data_list(graphs)
        return graphs
    

class AllInOneOrginal(nn.Module):
    def __init__(self, token_dim, token_num, cross_prune=0.1, inner_prune=0.3):
        super(AllInOneOrginal, self).__init__()
        self.prefix_prompt = True
        self.inner_prune = inner_prune
        self.cross_prune = cross_prune
        self.token_embeds = torch.nn.Parameter(torch.empty(token_num, token_dim))
        torch.nn.init.kaiming_uniform_(self.token_embeds, nonlinearity='leaky_relu', mode='fan_in', a=0.01)
        self.pg = self.pg_construct()

    @property
    def name(self,):
        return "all_in_one"

    def pg_construct(self,):
        token_sim = torch.mm(self.token_embeds, torch.transpose(self.token_embeds, 0, 1))
        token_sim = torch.sigmoid(token_sim)
        inner_adj = torch.where(token_sim < self.inner_prune, 0, token_sim)
        edge_index = inner_adj.nonzero().t().contiguous()
        pg = Data(x=self.token_embeds, edge_index=edge_index, y=torch.tensor([0]).long())
        return pg

    def forward(self, graph_batch, device = None):
        if isinstance(graph_batch, Batch):
            graph_batch = graph_batch.to_data_list()
        if device is not None:
            graph_batch = [graph.to(device) for graph in graph_batch]
        self.pg.to(graph_batch[0].x.device)
        inner_edge_index = self.pg.edge_index
        token_num = self.pg.x.shape[0]
        re_graph_batch = []
        for g in graph_batch:
            g_edge_index = g.edge_index + token_num
            cross_dot = torch.mm(self.pg.x, torch.transpose(g.x, 0, 1))
            cross_sim = torch.sigmoid(cross_dot)
            cross_adj = torch.where(cross_sim < self.cross_prune, 0, cross_sim)
            cross_edge_index = cross_adj.nonzero().t().contiguous()
            cross_edge_index[1] = cross_edge_index[1] + token_num
            x = torch.cat([self.pg.x, g.x], dim=0)
            edge_index = torch.cat([inner_edge_index, g_edge_index, cross_edge_index], dim=1)
            y = g.y
            data = Data(x=x, edge_index=edge_index, y=y)
            re_graph_batch.append(data)
        re_graph_batch = Batch.from_data_list(re_graph_batch)
        return re_graph_batch


class GPFPlus(nn.Module):
    def __init__(self, token_dim, token_num):
        super(GPFPlus, self).__init__()
        self.prefix_prompt = True
        self.token_embeds = torch.nn.Parameter(torch.empty(token_num, token_dim))
        torch.nn.init.kaiming_uniform_(self.token_embeds, nonlinearity='leaky_relu', mode='fan_in', a=0.01)
        self.dropout = nn.Dropout(0.3)

    @property
    def name(self,):
        return "gpf_plus"

    def forward(self, graph_batch, device = None):
        if isinstance(graph_batch, Batch):
            graph_batch = graph_batch.to_data_list()
        if device is not None:
            graph_batch = [graph.to(device) for graph in graph_batch]
        re_graph_batch = []
        for g in graph_batch:
            att_scores = F.softmax(g.x @ self.token_embeds.T, dim = 1)
            att_scores = self.dropout(att_scores)
            prompt = att_scores @ self.token_embeds
            x = g.x + prompt
            edge_index = g.edge_index
            y = g.y
            data = Data(x=x, edge_index=edge_index, y=y).to(g.x.device)
            re_graph_batch.append(data)
        re_graph_batch = Batch.from_data_list(re_graph_batch)
        return re_graph_batch


class GraphPrompt(nn.Module):
    def __init__(self, token_dim):
        super(GraphPrompt, self).__init__()
        self.prefix_prompt = False
        self.token_embeds = torch.nn.Parameter(torch.empty(1, token_dim))
        torch.nn.init.xavier_uniform_(self.token_embeds)

    @property
    def name(self,):
        return "graph_prompt"

    @classmethod
    def loss(cls, input:torch.Tensor, target:torch.Tensor, num_classes, temperature=1.0):
        centers, counts = cls.cal_temp_centers(input, target, num_classes)
        scores = F.cosine_similarity(input[:, None, :], centers[None, :, :], dim=-1) / temperature
        pos_scores = scores.gather(dim=1, index=target.view(-1, 1))
        neg_scores = scores.logsumexp(dim=1)
        loss = -(pos_scores - neg_scores).mean()
        return loss

    @staticmethod
    def cal_temp_centers(embeds, labels, num_classes):
        counts = labels.bincount(minlength=num_classes)
        centers = torch.zeros((num_classes, embeds.size(-1)), device=embeds.device)
        centers.scatter_reduce_(dim=0, index=labels.view(-1, 1).tile(1, embeds.size(-1)), src=embeds, reduce="sum")
        centers = centers / torch.maximum(counts, torch.tensor(1e-8))[:, None]
        return centers, counts

    def update_centers(self, pretrained_model, dataset, device):
        accumulated_centers = 0
        accumulated_counts = 0
        for i, (batch, idxs) in enumerate(dataset.train_loader):
            batch = batch.to(device)
            idxs = idxs.to(device)
            _, embeds = pretrained_model(
                batch,
                decoder = False,
                )
            embeds = self.forward(embeds, batch.batch)
            temp_centers, temp_counts = GraphPrompt.cal_temp_centers(embeds, batch.y, dataset.num_gclass)
            accumulated_centers += temp_centers * temp_counts.view(-1, 1)
            accumulated_counts += temp_counts.view(-1, 1)
        self._centers = accumulated_centers / accumulated_counts

    def get_centers(self,):
        return self._centers

    def forward(self, node_embeds, batch_idxs, device = None):
        node_embeds *= self.token_embeds
        graph_embeds = global_mean_pool(node_embeds, batch_idxs)
        return graph_embeds


class GraphPromptPlus(GraphPrompt):
    def __init__(self, in_channels, hidden_channels, num_layers):
        super(GraphPromptPlus, self).__init__(in_channels)
        self.prefix_prompt = False
        self.prompt_params = nn.ParameterList([torch.nn.Parameter(torch.empty(1, in_channels))])
        for _ in range(num_layers):
            self.prompt_params.append(torch.nn.Parameter(torch.empty(1, hidden_channels)))
        for param in self.prompt_params:
            torch.nn.init.xavier_uniform_(param)
        self.layer_weights = torch.nn.Parameter(torch.empty(len(self.prompt_params), 1))
        torch.nn.init.xavier_uniform_(self.layer_weights)

    @property
    def name(self,):
        return "graph_prompt_plus"

    def update_centers(self, pretrained_model, dataset, device):
        accumulated_centers = 0
        accumulated_counts = 0
        for i, (batch, idxs) in enumerate(dataset.train_loader):
            batch = batch.to(device)
            idxs = idxs.to(device)
            _, embeds = pretrained_model(
                batch,
                decoder = False,
                prompt_params = self.prompt_params
                )
            embeds = self.forward(embeds, batch.batch)
            temp_centers, temp_counts = GraphPrompt.cal_temp_centers(embeds, batch.y, dataset.num_gclass)
            accumulated_centers += temp_centers * temp_counts.view(-1, 1)
            accumulated_counts += temp_counts.view(-1, 1)
        self._centers = accumulated_centers / accumulated_counts
        
    def forward(self, node_embeds, batch_idxs, device = None):
        # ipdb.set_trace()
        n_nodes, n_feats = node_embeds[0].size()
        node_embeds[-1] *= self.prompt_params[-1]
        node_embeds = torch.cat([embds.view(1, -1) for embds in node_embeds], dim=0)
        weighted_embeds = node_embeds.T @ self.layer_weights
        graph_embeds = global_mean_pool(weighted_embeds.view(n_nodes, n_feats).contiguous(), batch_idxs)
        return graph_embeds
        

class GPPT(nn.Module):
    def __init__(self, in_channels, out_channels, num_centers):
        super(GPPT, self).__init__()
        self.prefix_prompt = True
        self.num_centers = num_centers
        self.out_channels = out_channels
        self.graph_conv = SAGEConv(in_channels, in_channels, root_weight=True)
        self.center_classifier = nn.Linear(in_channels, num_centers, bias=False)
        self.output_classifier = nn.ModuleList([nn.Linear(in_channels, out_channels, bias=False) for _ in range(num_centers)])

    @property
    def name(self,):
        return "gppt"

    def get_firsthop_ds_x(self, graphs, device = None):
        if isinstance(graphs, Batch):
            graphs = graphs.to_data_list()
        if device is not None:
            graphs = [g.to(device) for g in graphs]

        ds_x = []
        for g in graphs:
            x = self.graph_conv(g.x, edge_index)
            x = F.relu(x)
            one_hop_idxs = g.edge_index[1, (g.edge_index[0, :] == g.ego_idx).nonzero().view(-1)]
            ego_x = x[g.ego_idx]
            neighbor_x = x[one_hop_idxs].mean(dim=0)
            ds_x.append(torch.cat([ego_x, neighbor_x], dim=1))
        ds_x = torch.stack(ds_x, dim=0)

        return graphs, ds_x

    def get_ds_x(self, gnn_x, graph_batch, device = None, with_neighbor = False):
        if not isinstance(graph_batch, Batch):
            graph_batch = Batch.from_data_list(graph_batch)
        if device is not None:
            graph_batch = graph_batch.to(device)

        neighbor_counts = graph_batch.batch.unique(return_counts=True)[1]
        diff_idxs = (graph_batch.batch.roll(shifts=1) != graph_batch.batch).nonzero().view(-1)
        ego_idxs = diff_idxs + graph_batch.ego_idx
        ds_x = gnn_x[ego_idxs]
        
        if with_neighbor:
            batch_size = graph_batch.batch.max() + 1
            graph_batch.batch[ego_idxs] = batch_size
            neighbor_x = torch.zeros((batch_size, gnn_x.size(-1)))
            neighbor_x.scatter_add_(dim=0, index=graph_batch.batch[:, None].tile(gnn_x.size(-1)), src=gnn_x)
            neighbor_x = neighbor_x[:-1]
            ds_x = torch.cat([ds_x, neighbor_x], dim=1)

        return ds_x
    
    def init_weights(self, graphs, device = None):
        x = self.graph_conv(g.x, edge_index)
        x = F.relu(x)
        ds_x = self.get_ds_x(graphs, device)
        _, embeds = pretrained_model(
            batch,
            decoder = False,
            device = self.device
            )
        
        cluster = KMeans(n_clusters=self.num_centers, random_state=0).fit(ds_x.detach().cpu().numpy())
        cluster_centers_x = torch.FloatTensor(cluster.cluster_centers_).to(ds_x.device)
        self.center_classifier.weight.data.copy_(cluster_centers_x)

        class_centers = torch.zeros((self.out_channels, ds_x.size(-1)))
        labels = torch.tensor([g.y for g in graphs]).to(ds_x.device)
        class_centers.scatter_reduce_(dim=1, index=labels.view(-1, 1).tile(1, ds_x.size(-1)), src=ds_x, reduce="mean")

        for layer in self.output_classifier:
            layer.weight.data.copy_(temp)

    def update_centers(self, x):
        device = x.device
        cluster = KMeans(n_clusters=self.num_centers,random_state=0).fit(x.detach().cpu().numpy())
        cluster_centers_x = torch.FloatTensor(cluster.cluster_centers_).to(device)
        self.center_classifier.weight.data.copy_(cluster_centers_x)

    def get_reg_loss(self,):
        norm_sum = 0
        p_count = 0
        for name, p in self.named_parameters():
            if name.startswith('output_classifier'):
                norm_sum += torch.norm(p@p.T -torch.eye(p.size(0), device=p.device))
                p_count += 1
        return 0.001 * norm_sum/p_count

    def forward(self, gnn_x, graphs, device = None):
        ds_x = self.get_ds_x(gnn_x, graphs, device)
        self.mid_x = ds_x

        center_scores = self.center_classifier(ds_x)
        center_idx = center_scores.argmax(dim=1)
        logits = torch.zeros(ds_x.size(0), self.out_channels, device=ds_x.device)
        for i, layer in enumerate(self.output_classifier):
            mask = center_idx==i
            logits[mask] = layer(ds_x[mask])

        return logits


class HeadTuning(nn.Module):
    def __init__(self, *args, **kwargs):
        super(HeadTuning, self).__init__()
        pass

    @property
    def name(self,):
        return "head_tuning"

    def forward(self, graph_batch, device = None):
        if device is not None:
            graph_batch = graph_batch.to(device)
        return graph_batch