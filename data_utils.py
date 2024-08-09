import torch, random, os
import numpy as np
from torch_geometric.datasets import QM9, TUDataset, CitationFull, Planetoid, Airports
from utils import *
from model import *
import pandas as pd
from torch_geometric.loader import DataLoader as PyG_Dataloader
from torch_geometric.data import Data, Batch, Dataset as PyG_Dataset
from torch_geometric.utils import k_hop_subgraph, subgraph, dense_to_sparse, homophily, degree, to_networkx, get_ppr, is_undirected
from torch_geometric.loader import NeighborLoader
from torch.utils.data import Dataset, DataLoader, Sampler
from sklearn.datasets import make_spd_matrix
from sklearn.mixture import GaussianMixture
from collections import OrderedDict
from copy import deepcopy
from sklearn.utils import shuffle as sk_shuffl
from networkx import pagerank, diameter, all_pairs_shortest_path, density, clustering
import ipdb
import gc


def graph_collate(batch):
    g_list = []
    idxs = []
    if not isinstance(batch, list):
        for g in batch:
            g_list.extend(g[0])
            idxs.append(g[1])
    else:
        for d in batch:
            g_list.append(d[0])
            idxs.append(d[1])
    g_batch = Batch.from_data_list(g_list)
    idxs = torch.as_tensor(idxs)
    return g_batch, idxs


def add_multivariate_noise(features, mean_shift, cov_scale) -> None:
    n_samples, n_feats = features.size()
    mean = np.ones(n_feats) * mean_shift
    cov_matrix = np.eye(n_feats) * cov_scale
    noise = np.random.multivariate_normal(mean, cov_matrix, n_samples)
    features += torch.as_tensor(noise, dtype=torch.float, device=features.device)
    return features


def graph_ds_add_noise(dataset, mean_shift = None, cov_scale = None):
    x_idxs = dataset.slices["x"]
    for class_id in range(dataset.num_classes):
        if mean_shift == None:
            mean_shift = np.random.uniform(-2, 2)
        else:
            mean_shift = np.random.uniform(-mean_shift, mean_shift)
        cov_scale = 1. if cov_scale is None else cov_scale
        temp_idxs = (dataset.y == class_id).nonzero().T[0]
        temp_idxs = torch.cat((x_idxs[temp_idxs][:, None], x_idxs[temp_idxs+1][:, None]), dim=1)
        class_idxs = []
        for i in range(temp_idxs.size(0)):
            class_idxs.append(torch.arange(temp_idxs[i, 0], temp_idxs[i, 1]))
        class_idxs = torch.cat(class_idxs)
        dataset.x[class_idxs, :] = add_multivariate_noise(dataset.x[class_idxs, :], mean_shift, cov_scale)
    return dataset


class DomianShift():
    def __init__(self, n_domains=2, *args, **kwargs) -> None:
        self.n_domains = n_domains

    @classmethod
    def save_to_file(cls, dataset_name, domain_idx_dict, dir_path=None):
        if dir_path is None:
            dir_path = f"./files/{dataset_name}/domain_indicies"
        os.makedirs(dir_path, exist_ok=True)
        files_path = os.path.join(dir_path, f"domain_idicies.txt")
        with open(files_path, "w") as f_handle:
            for domain_id, idxs in domain_idx_dict.items():
                line = f"{domain_id}: "
                line += " ".join(idxs.cpu().numpy().astype(str).tolist())
                f_handle.write(line + "\n")
        return files_path

    @classmethod
    def load_from_file(cls, filename_):
        with open(filename_, "r") as f_handle:
            domain_idx_dict = dict()
            for line in f_handle.readlines():
                domain_id, idxs = line.strip().split(":")
                domain_idx_dict[int(domain_id)] = torch.as_tensor(np.array(idxs.strip(" ").split(" ")).astype(np.int64))
        return domain_idx_dict


class GDataset(nn.Module):
    def __init__(self,):
        pass

    def init_loaders_(self,):
        "Implement this is children classes"
        pass

    def normalize_feats_(self,):
        "Implement this is children classes"
        return "x"

    def reset_preds(self,):
        self.train_ds._reset_preds()
        
    def update_preds(self, idxs, preds):
        self.train_ds._update_preds(idxs, preds)

    def get_preds(self,):
        return self.train_ds._get_preds()

    def init_ds_idxs_(self, train_idxs, valid_idxs, test_idxs, train_test_split, label_reduction, shuffle, seed):
        if (train_idxs is not None) and (valid_idxs is not None) and (test_idxs is not None):
            self.n_train = train_idxs.size(0)
            self.n_valid = valid_idxs.size(0)
            self.n_test = test_idxs.size(0)
            self.train_idxs = train_idxs[:int(self.n_train * (1-label_reduction))]
            self.valid_idxs = valid_idxs
            self.test_idxs = test_idxs
        else:
            all_idxs = torch.arange(self.num_gsamples)
            if shuffle: 
                fix_seed(seed)
                perm = torch.randperm(self.num_gsamples)
                all_idxs = all_idxs[perm]
            if train_test_split[0] + train_test_split[1] != 1.0:
                valid_per = 1 - (train_test_split[0] + train_test_split[1])
            else:
                valid_per = 0.0
            self.all_idxs = all_idxs
            self.n_train = int(self.num_gsamples * train_test_split[0])
            self.n_valid = int(self.num_gsamples * valid_per)
            self.n_test = self.num_gsamples - (self.n_train + self.n_valid)
            self.train_idxs = all_idxs[:int(self.n_train * (1-label_reduction))]
            self.valid_idxs = all_idxs[self.n_train:self.n_train + self.n_valid]
            self.test_idxs = all_idxs[self.n_train + self.n_valid:self.n_train + self.n_valid + self.n_test]

    def initialize(self,):
        "Implement this is children classes"
        return "x"


class SimpleDataset(Dataset):
    def __init__(self,
                 graph_list: List,
                 **kwargs) -> None:
        super(SimpleDataset, self).__init__()
        self._data = graph_list
        self.preds = torch.ones((len(graph_list),)).long() * -1

    def _reset_preds(self,):
        self.preds = torch.ones_like(self.preds).long() * -1

    def _update_preds(self, idxs, preds):
        self.preds[idxs] = preds

    def _get_preds(self):
        return self.preds

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx], idx


class SubsetSampler(Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


class SubsetRandomSampler(SubsetSampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in torch.randperm(len(self.indices)))


def node_property_splits(data_graph, src_ratio, shift_mode="pr", select_mode="soft", drop_unconnected = True):
    n_samples = data_graph.x.size(0)
    print(shift_mode)
    if shift_mode == "pr":
        property_dict = pagerank(to_networkx(data_graph))
    elif shift_mode == "cc":
        property_dict = clustering(to_networkx(data_graph))
    else:
        raise Exception("Shift mode is not supported!")
    properties = torch.zeros((n_samples,))
    for key, value in property_dict.items():
        properties[key] = value
    if select_mode == "soft":
        selec_probs = properties / properties.sum()
        src_idxs = torch.as_tensor(np.random.choice(n_samples, int(n_samples * src_ratio), replace=False, p=selec_probs.numpy()))
    else:
        _, sorted_idxs = properties.sort(descending=False)
        src_idxs = sorted_idxs[int(n_samples * src_ratio):]

    if drop_unconnected:
        assert is_undirected(data_graph.edge_index)
        sub_edges, _ = subgraph(src_idxs, data_graph.edge_index)
        assert is_undirected(sub_edges)
        connected_idxs = sub_edges[0, :].unique()
        src_idxs = src_idxs[torch.isin(src_idxs, connected_idxs)]

    src_idxs = src_idxs.sort().values
    src_mask = torch.zeros((n_samples,), dtype=bool)
    src_mask[src_idxs] = True
    tgt_idxs = (~src_mask).nonzero().T[0]

    return src_idxs, tgt_idxs


def merge_induced_graphs(ref_idxs, all_nids, nids, edges):
    mapping = dict()
    new_nids = dict()
    new_edges = dict()
    last_id = torch.cat(list(nids.values())).max()
    for i in range(len(nids)):
        temp_nids = all_nids[ref_idxs[i].item()]
        unmatched_idxs = temp_nids[~torch.isin(temp_nids, ref_idxs[nids[i]])]
        n_unmatched = unmatched_idxs.size(0)
        mapped_unmachted = torch.ones_like(unmatched_idxs) * -1
        ego_arg = (nids[i] == i).nonzero().T[0]

        for j, k in enumerate(unmatched_idxs.tolist()):
            if k not in mapping:
                mapping[k] = last_id
                last_id = last_id + 1
            mapped_unmachted[j] = mapping[k]

        new_edge_tgts = torch.arange(n_unmatched) + nids[i].size(0)
        temp_edges = torch.cat((ego_arg.tile(1, n_unmatched), new_edge_tgts[None, :]), dim=0)
        new_edges[i] = torch.cat((edges[i], temp_edges, temp_edges[[1, 0], :]), dim=1)
        new_nids[i] = torch.cat((nids[i], mapped_unmachted))

        assert new_edges[i].max()+1 == new_nids[i].size(0)
        assert is_undirected(edges[i])
    return new_nids, new_edges, mapping


def get_induced_graphs(dataset, n_hops, smallest_size=None, largest_size=None, max_hop=5):
    induced_nodes = dict()
    induced_edge_idxs = dict()
    for index in range(dataset.x.size(0)):

        current_hop = n_hops
        subset, edges, _, _ = k_hop_subgraph(
            node_idx=index, num_hops=current_hop, edge_index = dataset.edge_index,
            num_nodes = dataset.x.size(0), relabel_nodes = True
            )

        if smallest_size:
            while len(subset) < smallest_size and current_hop < max_hop:
                current_hop += 1
                subset, _, _, _ = k_hop_subgraph(
                    node_idx = index, num_hops=current_hop,
                    edge_index = dataset.edge_index,
                    num_nodes = dataset.x.size(0), relabel_nodes = True
                    )
    
            if len(subset) < smallest_size:
                need_node_num = smallest_size - len(subset)
                pos_nodes = torch.argwhere(dataset.y == int(current_label))
                candidate_nodes = torch.from_numpy(np.setdiff1d(pos_nodes.numpy(), subset.numpy()))
                candidate_nodes = candidate_nodes[torch.randperm(candidate_nodes.size(0))][:need_node_num]
                subset = torch.cat([torch.flatten(subset), torch.flatten(candidate_nodes)])

        if largest_size:
            if len(subset) > largest_size:
                subset = subset[torch.randperm(subset.shape[0])][:largest_size - 1]
                subset = torch.unique(torch.cat([torch.LongTensor([index]), torch.flatten(subset)]))

        if smallest_size or largest_size:
            edges, _ = subgraph(subset, dataset.edge_index, num_nodes=dataset.x.size(0), relabel_nodes=True)
            
        induced_nodes[index] = subset
        induced_edge_idxs[index] = edges

    return induced_nodes, induced_edge_idxs


def get_nids(num_nodes, all_nids, ego_idxs):
    nids = [all_nids[idx.item()] for idx in ego_idxs]
    all_idxs = torch.cat(nids).unique()
    mask = torch.ones((num_nodes,), dtype=int)
    mask[all_idxs] = 0
    diffs = mask.cumsum(dim=0)
    for i, nid in enumerate(nids):
        nids[i] = torch.as_tensor([j-diffs[j] for j in nid])
    return nids, all_idxs


class InducedDataset(Dataset):
    def __init__(self, node_ids, edge_idxs, x, y, **kwargs) -> None:
        super(InducedDataset, self).__init__()
        self.x = x
        self.all_nids = node_ids
        self.all_edges = edge_idxs
        self.y = y
        self.preds = torch.ones_like(self.y).long() * -1

    def _reset_preds(self,):
        self.preds = torch.ones_like(self.preds).long() * -1

    def _update_preds(self, idxs, preds):
        self.preds[idxs] = preds

    def _get_preds(self):
        return self.preds
        
    def _copy_idxs(self, ego_idxs):
        nids, all_idxs = get_nids(self.x.size(0), self.all_nids, ego_idxs)
        induced_ds = InducedDataset(
            nids,
            [self.all_edges[idx.item()] for idx in ego_idxs],
            self.x[all_idxs],
            self.y[ego_idxs]
        )
        return induced_ds

    def __len__(self):
        return len(self.all_nids)

    def __getitem__(self, idx):
        x = self.x[self.all_nids[idx]]
        y = self.y[idx]
        edges = self.all_edges[idx]
        return Data(x=x, edge_index=edges, y=y), idx


class NodeToGraphDataset(GDataset):
    def __init__(self,
                 main_data,
                 all_idxs,
                 all_nids,
                 all_edges,
                 ego_idxs,
                 n_hopes = 2,
                 **kwargs) -> None:
        super(NodeToGraphDataset, self).__init__()
        self._data = InducedDataset( 
            all_nids, 
            all_edges,
            main_data.x[all_idxs],
            main_data.y[ego_idxs]
        )
        self.n_feats = main_data.x.size(1)
        self.num_nsamples = main_data.x.size(0)
        self.num_nclass = main_data.y.unique().size(0)
        self.num_gclass = self.num_nclass
        self.num_gsamples = len(self._data)
        self.n_hopes = n_hopes

    @property
    def x(self,):
        return self._data.x

    def normalize_feats_(self, normalize_mode, **kwargs):
        if self.base_ds is not None:
            _, train_normal_params = normalize_(self.base_ds.x, dim=0, mode=normalize_mode)
            self.train_ds.x, _ = normalize_(self.train_ds.x, dim=0, mode=normalize_mode, normal_params = train_normal_params)
        else:
            self.train_ds.x, train_normal_params = normalize_(self.train_ds.x, dim=0, mode=normalize_mode)
        if self.n_valid > 0:
            self.valid_ds.x, _ = normalize_(
                self.valid_ds.x, dim=0, 
                mode=normalize_mode, normal_params = train_normal_params)
        if self.n_test > 0:
            self.test_ds.x, _ = normalize_(
                self.test_ds.x, dim=0, 
                mode=normalize_mode, normal_params = train_normal_params)

    def init_loaders_(self, batch_size, loader_collate = graph_collate):
        self.train_loader = DataLoader(self.train_ds, batch_size=batch_size, shuffle=True, collate_fn=loader_collate, num_workers=1)
        self.valid_loader = DataLoader(self.valid_ds, batch_size=batch_size, shuffle=False, collate_fn=loader_collate, num_workers=1)
        self.test_loader = DataLoader(self.test_ds, batch_size=batch_size, shuffle=False, collate_fn=loader_collate, num_workers=1)

    def initialize(
            self,
            train_idxs: torch.Tensor = None,
            valid_idxs: torch.Tensor = None,
            test_idxs: torch.Tensor = None,
            train_test_split = [0.85, 0.15],
            loader_collate = graph_collate,
            batch_size = 32, 
            normalize_mode = None,
            shuffle = False, 
            label_reduction = 0.0, 
            **kwargs) -> None:
        self.init_ds_idxs_(
            train_idxs = train_idxs, valid_idxs = valid_idxs, test_idxs = test_idxs,
            train_test_split = train_test_split,
            label_reduction = label_reduction,
            shuffle = shuffle, seed = kwargs["seed"] if "seed" in kwargs else 2411
        )
        if label_reduction > 0.0:
            self.base_ds = self._data._copy_idxs(self.all_idxs[:self.n_train])
        else:
            self.base_ds = None
        self.train_ds = self._data._copy_idxs(self.train_idxs)
        self.valid_ds = self._data._copy_idxs(self.valid_idxs)
        self.test_ds = self._data._copy_idxs(self.test_idxs)
        if normalize_mode is not None:
            self.normalize_feats_(normalize_mode)
        self.init_loaders_(batch_size, loader_collate)


def graph_property_splits(dataset, src_ratio, shift_mode="homophily", select_mode="soft", n_node_cls=None):
    n_graphs = len(dataset)
    property_ratios = torch.zeros((n_graphs,), dtype = torch.float)
    print(shift_mode)
    for i, graph in enumerate(dataset):
        if shift_mode == "homophily":
            # same_edge = (graph.x[graph.edge_index[0, :], -n_node_cls:] * graph.x[graph.edge_index[1, :], -n_node_cls:]).sum()
            # homophily_ratios[i] = same_edge / max(1, graph.edge_index.size(1))
            y = graph.x[:, -n_node_cls:].argmax(dim=1)
            property_ratios[i] = homophily(graph.edge_index, y, method="edge")
        elif shift_mode == "diameter":
            all_paths = all_pairs_shortest_path(to_networkx(graph))
            max_diameter = 0
            for outer_key, outer_value in dict(all_paths).items():
                for inner_key, inner_value in outer_value.items():
                    max_diameter = max(len(inner_value)-1, max_diameter)
            property_ratios[i] = max_diameter
        elif shift_mode == "density":
            property_ratios[i] = density(to_networkx(graph))
        else:
            raise Exception("Shift mode is not supported.")
    if select_mode == "soft":
        selec_probs = property_ratios / property_ratios.sum()
        src_idxs = torch.as_tensor(np.random.choice(n_graphs, int(n_graphs * src_ratio), replace=False, p=selec_probs.numpy()))
    else:
        _, sorted_idxs = property_ratios.sort(descending=False)
        src_idxs = sorted_idxs[int(n_graphs * src_ratio):]
    src_idxs = src_idxs.sort().values
    select_mask = torch.zeros((n_graphs,), dtype=bool)
    select_mask[src_idxs] = True
    tgt_idxs = (~select_mask).nonzero().T[0]
    return src_idxs, tgt_idxs


class FromPyGGraph(GDataset):
    def __init__(self,
                 main_dataset: PyG_Dataset,
                 **kwargs) -> None:
        super(FromPyGGraph, self).__init__()
        self._data = deepcopy(main_dataset)
        self.n_feats = self._data.x.size(1)
        self.num_nsamples = self._data.x.size(0)
        self.num_nclass = self._data.num_node_labels
        self.num_gclass = self._data.num_classes
        self.num_gsamples = len(self._data)
        class_weight = self.num_gsamples / (self.num_gclass * torch.bincount(self._data.y))
        self.class_weight = torch.as_tensor(class_weight, dtype=torch.float)

    def gen_graph_ds(self, dataset):
        if len(dataset) == 0:
            return []
        x_idxs = dataset.slices["x"]
        all_graphs = []
        for i in range(x_idxs.size(0)-1):
            g = dataset[i]
            temp_g = Data(
                x = dataset._data.x[x_idxs[i]:x_idxs[i+1], :], 
                edge_index = g.edge_index, y = g.y
            )
            all_graphs.append(temp_g)
        return all_graphs

    def normalize_feats_(self, normalize_mode, **kwargs):
        if self.base_ds is not None:
            _, train_normal_params = normalize_(self.base_ds._data.x, dim=0, mode=normalize_mode)
            self.train_ds._data.x, _ = normalize_(self.train_ds._data.x, dim=0, mode=normalize_mode, normal_params = train_normal_params)
        else:
            self.train_ds._data.x, train_normal_params = normalize_(self.train_ds._data.x, dim=0, mode=normalize_mode)
        if self.n_valid > 0:
            self.valid_ds._data.x, _ = normalize_(self.valid_ds._data.x, dim=0, mode=normalize_mode, normal_params = train_normal_params)
        if self.n_test > 0:
            self.test_ds._data.x, _ = normalize_(self.test_ds._data.x, dim=0, mode=normalize_mode, normal_params = train_normal_params)
        
    def init_loaders_(self, batch_size, loader_collate = graph_collate):            
        self.train_loader = DataLoader(self.train_ds, batch_size=batch_size, shuffle=True, collate_fn=loader_collate, num_workers=1)
        self.valid_loader = DataLoader(self.valid_ds, batch_size=batch_size, shuffle=False, collate_fn=loader_collate, num_workers=1)
        self.test_loader = DataLoader(self.test_ds, batch_size=batch_size, shuffle=False, collate_fn=loader_collate, num_workers=1)
        
    def initialize(
            self,
            train_idxs: torch.Tensor = None,
            valid_idxs: torch.Tensor = None,
            test_idxs: torch.Tensor = None,
            train_test_split = [0.85, 0.15],
            loader_collate = graph_collate,
            batch_size = 32, 
            normalize_mode = None,
            shuffle = False, 
            label_reduction = 0.0, 
            **kwargs) -> None:
        self.init_ds_idxs_(
            train_idxs = train_idxs, valid_idxs = valid_idxs, test_idxs = test_idxs,
            train_test_split = train_test_split,
            label_reduction = label_reduction,
            shuffle = shuffle, seed = kwargs["seed"] if "seed" in kwargs else 2411
        )
        if label_reduction > 0.0:
            self.base_ds = self._data.copy(self.all_idxs[:self.n_train])
        else:
            self.base_ds = None
        self.train_ds = self._data.copy(self.train_idxs)
        self.valid_ds = self._data.copy(self.valid_idxs) if self.n_valid > 0 else []
        self.test_ds = self._data.copy(self.test_idxs) if self.n_test > 0 else []
        if normalize_mode is not None:
            self.normalize_feats_(normalize_mode)
        self.train_ds = SimpleDataset(self.gen_graph_ds(self.train_ds))
        self.valid_ds = SimpleDataset(self.gen_graph_ds(self.valid_ds))
        self.test_ds = SimpleDataset(self.gen_graph_ds(self.test_ds))
        self.init_loaders_(batch_size, loader_collate)
        

class GenDataset(object):
    def __init__(self, logger) -> None:
        self.logger = logger

    def get_node_dataset(
        self,
        ds_name,
        shift_type = "structural",
        p_intra = 0.0,
        p_inter = 0.0,
        cov_scale = 2,
        mean_shift = 0,
        shift_mode = "pr",
        s_split = [0.8, 0.2],
        t_split = [0.8, 0.2],
        src_ratio = 0.5,
        batch_size = 32,
        n_hopes = 2,
        norm_mode = "max",
        node_attributes = True,
        label_reduction = 0.0,
        seed = 2411,
        select_mode = "soft"
    ):

        """ Currently supported datasets:
            - Cora_
        """
        dataset = Planetoid(
            root = f'data/{ds_name}',
            name = ds_name
            )
        data = deepcopy(dataset._data.subgraph(dataset._data.edge_index.unique()))

        fix_seed(seed)
        all_nids, all_edges = get_induced_graphs(data, n_hops=1)
        src_ego_idxs, tgt_ego_idxs = node_property_splits(data, src_ratio, shift_mode, select_mode, drop_unconnected=False)
        s_data = deepcopy(data.subgraph(src_ego_idxs))
        t_data = deepcopy(data.subgraph(tgt_ego_idxs))
        src_nids, src_edges = get_induced_graphs(s_data, n_hops=2)
        tgt_nids, tgt_edges = get_induced_graphs(t_data, n_hops=2)

        src_nids, src_edges, src_mapping = merge_induced_graphs(src_ego_idxs, all_nids, src_nids, src_edges)
        tgt_nids, tgt_edges, tgt_mapping = merge_induced_graphs(tgt_ego_idxs, all_nids, tgt_nids, tgt_edges)
        
        def get_all_idxs(ego_idxs, aux_mapping):
            aux_mapping = torch.as_tensor(list(aux_mapping.items()))
            aux_mapping = aux_mapping[aux_mapping[:, 1].sort().indices, :]
            assert (~torch.isin(ego_idxs, aux_mapping[:, 0])).all()
            ego_all_idxs = torch.cat((ego_idxs, aux_mapping[:, 0]))
            return ego_all_idxs
        
        src_all_idxs = get_all_idxs(src_ego_idxs, src_mapping)
        tgt_all_idxs = get_all_idxs(tgt_ego_idxs, tgt_mapping)

        s_dataset = NodeToGraphDataset(
            data,
            src_all_idxs,
            src_nids,
            src_edges,
            src_ego_idxs
        )
        s_dataset.initialize(
            train_test_split = s_split,
            batch_size = batch_size,
            normalize_mode = norm_mode,
            shuffle = True,
            label_reduction = 0.0,
            seed = seed
        )
        t_dataset = NodeToGraphDataset(
            data,
            tgt_all_idxs,
            tgt_nids,
            tgt_edges,
            tgt_ego_idxs
        )
        t_dataset.initialize(
            train_test_split = t_split,
            batch_size = batch_size,
            normalize_mode = norm_mode,
            shuffle = True,
            label_reduction = label_reduction,
            seed = seed
        )
        
        return s_dataset, t_dataset


    def get_graph_dataset(
        self,
        ds_name,
        shift_type = "structural",
        p_intra = 0.0,
        p_inter = 0.0,
        cov_scale = 2,
        mean_shift = 0,
        shift_mode = "homophily",
        store_to_path = "./data",
        s_split = [0.8, 0.2],
        t_split = [0.8, 0.2],
        src_ratio = 0.5,
        batch_size = 32,
        norm_mode = "max",
        node_attributes = True,
        label_reduction = 0.0,
        seed = 2411,
        select_mode = "soft"
    ):
        
        """ Currently supported datasets: 
            - ENZYMES
            - PROTEINS_full
        """
        dataset = TUDataset(
            root = store_to_path,
            name = ds_name,
            use_node_attr = node_attributes
        )

        n_node_cls = dataset.num_node_labels

        fix_seed(seed)
        if shift_type == "structural":
            src_idxs, tgt_idxs = graph_property_splits(dataset, src_ratio, shift_mode, select_mode, n_node_cls)
        else:
            ntotal_graphs = len(dataset)
            perm = torch.randperm(ntotal_graphs)
            src_idxs = perm[:int(ntotal_graphs * src_ratio)]
            tgt_idxs = perm[int(ntotal_graphs * src_ratio):]

        domain_idx_dict = {0:src_idxs, 1:tgt_idxs}
        DomianShift.save_to_file(ds_name, domain_idx_dict)
        s_ds = dataset.copy(src_idxs)
        t_ds = dataset.copy(tgt_idxs)

        s_dataset = FromPyGGraph(s_ds)
        s_dataset.initialize(
            train_test_split = s_split,
            batch_size = batch_size,
            normalize_mode = norm_mode,
            shuffle = True,
            label_reduction = 0.0,
            seed = seed
        )

        if shift_type == "feature":
            if shift_mode == "class_wise":
                t_ds = graph_ds_add_noise(t_ds, mean_shift, cov_scale)
            else:
                t_ds.x[torch.arange(t_ds.x.size(0)), :] = add_multivariate_noise(
                    t_ds.x[torch.arange(t_ds.x.size(0)), :], mean_shift, cov_scale
                )
        if  shift_type not in ["feature", "structural"]:
            raise Exception("Shift type is not supported!")
            
        t_dataset = FromPyGGraph(t_ds)
        t_dataset.initialize(
            train_test_split = t_split,
            batch_size = batch_size,
            normalize_mode = norm_mode,
            shuffle = True,
            label_reduction = label_reduction,
            seed = seed
        )
        return s_dataset, t_dataset