import torch, os
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
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
    
    def forward(self, graph_batch, decoder = True, device = None):
        if isinstance(graph_batch, list):
            graph_batch = Batch.from_data_list(graph_batch)
        if device is not None:
            graph_batch = graph_batch.to(device)
        if not decoder:
            scores = self.decoder(graph_batch)
            return scores, "embeds"
        x = graph_batch.x
        for i, layer in enumerate(self.gnn_layers):
            x = layer(x, graph_batch.edge_index)
            x = self.bns[i](x) if self.with_bn else x
            x = F.relu(x)
            if i < len(self.gnn_layers) - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)
        if not self.with_head:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.gnn_decoder(x, graph_batch.edge_index)
            scores = global_mean_pool(x, graph_batch.batch)
        if self.with_head:
            x = global_mean_pool(x, graph_batch.batch)
            scores = F.dropout(x, p=self.dropout, training=self.training)
            scores = self.linear_decoder(scores)
        return scores, x
    

class BasePrompt(nn.Module):
    def __init__(
            self,
            emb_dim,
            h_dim,
            output_dim,
            prompt_fn = "add_tokens",
            token_num = 30,
            cross_prune=0.1, 
            inner_prune=0.3,
            attn_dropout=0.3,
            input_dropout=0.3,
            attn_with_param=False
        ) -> None:
        super(BasePrompt, self).__init__()
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
        self.inner_prune = inner_prune
        self.cross_prune = cross_prune
        self.token_embeds = torch.nn.Parameter(torch.empty(token_num, token_dim))
        torch.nn.init.kaiming_uniform_(self.token_embeds, nonlinearity='leaky_relu', mode='fan_in', a=0.01)
        self.pg = self.pg_construct()

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
        if not isinstance(re_graph_batch, Batch):
            assert isinstance(re_graph_batch, List)
            re_graph_batch = Batch.from_data_list(re_graph_batch)
        return re_graph_batch


class AllInOneModified(nn.Module):
    def __init__(self, token_dim, token_num, cross_prune=0.1, inner_prune=0.3):
        super(AllInOneModified, self).__init__()
        self.inner_prune = inner_prune
        self.cross_prune = cross_prune
        self.token_embeds = torch.nn.Parameter(torch.empty(token_num, token_dim))
        torch.nn.init.kaiming_uniform_(self.token_embeds, nonlinearity='leaky_relu', mode='fan_in', a=0.01)
        self.pg = self.pg_construct()

    def pg_construct(self,):
        token_sim = self.token_embeds @ self.token_embeds.T
        token_sim.fill_diagonal_(.0)
        token_sim = torch.sigmoid(token_sim)
        inner_adj = torch.where(token_sim < self.inner_prune, 0, token_sim)
        edge_index = inner_adj.nonzero().T.contiguous()
        pg = Data(x=self.token_embeds, edge_index=edge_index, y=torch.tensor([0]).long())
        return pg

    def get_inner_edges(self, x):
        token_sim = x @ x.T
        token_sim.fill_diagonal_(.0)
        token_sim = torch.sigmoid(token_sim)
        inner_adj = torch.where(token_sim < self.inner_prune, 0, token_sim)
        edge_index = inner_adj.nonzero().T.contiguous()
        return edge_index

    def forward(self, graph_batch, device = None):
        if isinstance(graph_batch, Batch):
            graph_batch = graph_batch.to_data_list()
        if device is not None:
            graph_batch = [graph.to(device) for graph in graph_batch]
        # self.pg = self.pg_construct()
        self.pg.to(graph_batch[0].x.device)
        re_graph_batch = []
        for g in graph_batch:
            cross_dot = torch.mm(self.pg.x, torch.transpose(g.x, 0, 1))
            cross_sim = torch.sigmoid(cross_dot)  # 0-1 from prompt to input graph
            cross_adj = torch.where(cross_sim < self.cross_prune, 0, cross_sim)
            cross_edge_index = cross_adj.nonzero().T.contiguous()                
            added_tokens = cross_edge_index[0].unique()
            x = torch.cat([self.pg.x[added_tokens], g.x], dim=0)
            g_edge_index = g.edge_index + added_tokens.size(0)
            cross_edge_index[0] = (added_tokens[None, :] == cross_edge_index[0][:, None]).nonzero()[:, 1]
            cross_edge_index[1] = cross_edge_index[1] + added_tokens.size(0)
            inner_edge_index = self.get_inner_edges(self.pg.x[added_tokens])
            edge_index = torch.cat([inner_edge_index, g_edge_index, cross_edge_index], dim=1)
            y = g.y
            data = Data(x=x, edge_index=edge_index, y=y).to(graph_batch[0].x.device)
            re_graph_batch.append(data)
        if not isinstance(re_graph_batch, Batch):
            assert isinstance(re_graph_batch, List)
            re_graph_batch = Batch.from_data_list(re_graph_batch)
        return re_graph_batch


class GPFPlus(nn.Module):
    def __init__(self, token_dim, token_num):
        super(GPFPlus, self).__init__()
        self.token_embeds = torch.nn.Parameter(torch.empty(token_num, token_dim))
        torch.nn.init.kaiming_uniform_(self.token_embeds, nonlinearity='leaky_relu', mode='fan_in', a=0.01)
        self.dropout = nn.Dropout(0.3)

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
            data = Data(x=x, edge_index=edge_index, y=y).to(graph_batch[0].x.device)
            re_graph_batch.append(data)
        if not isinstance(re_graph_batch, Batch):
            assert isinstance(re_graph_batch, List)
            re_graph_batch = Batch.from_data_list(re_graph_batch)
        return re_graph_batch


class OrgGCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(OrgGCN, self).__init__()
        self.gc1 = OrgGraphConvolution(nfeat, nhid)
        self.gc2 = OrgGraphConvolution(nhid, nhid)
        self.bn1 = nn.BatchNorm1d(nhid)
        self.bn2 = nn.BatchNorm1d(nhid)
        self.head = nn.Linear(nhid, nclass)
        self.dropout = dropout

    def forward(self, x_adj_list, decoder=True):
        if not decoder:
            scores = scores = self.head(x_adj_list)
            return scores, ""
        g_embeds = []
        for i, (x, adj) in enumerate(x_adj_list):
            x = self.gc1(x, adj)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.gc2(x, adj)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
            g_embeds.append(x.mean(dim=0))
        g_embeds = torch.stack(g_embeds)
        scores = self.head(g_embeds)
        return scores, g_embeds


class OrgGraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, bias=True):
        super(OrgGraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / np.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input_x, adj):
        support = torch.mm(input_x, self.weight)
        output = torch.sparse.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'