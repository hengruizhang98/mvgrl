from utils import preprocess_features, normalize_adj
from sklearn.preprocessing import MinMaxScaler
from utils import compute_ppr
import scipy.sparse as sp
import networkx as nx
import numpy as np
import os
import dgl
from dgl.data import CoraGraphDataset, CiteseerGraphDataset
import torch as th

def process_dataset(name):
    if name == 'cora':
        dataset = CoraGraphDataset()
    elif name == 'citeseer':
        dataset = CiteseerGraphDataset()

    graph = dataset[0]
    feat = graph.ndata.pop('feat')
    label = graph.ndata.pop('label')

    train_mask = graph.ndata.pop('train_mask')
    val_mask = graph.ndata.pop('val_mask')
    test_mask = graph.ndata.pop('test_mask')

    train_idx = th.nonzero(train_mask, as_tuple=False).squeeze()
    val_idx = th.nonzero(val_mask, as_tuple=False).squeeze()
    test_idx = th.nonzero(test_mask, as_tuple=False).squeeze()

    nx_g = dgl.to_networkx(graph)

    diff_adj = compute_ppr(nx_g, 0.2)
    diff_adj[diff_adj < 0.01] = 0

    # if name == 'citeseer':
    #     feat = preprocess_features(feat)
    #     epsilons = [1e-5, 1e-4, 1e-3, 1e-2]
    #     avg_degree = graph.number_of_edges() / graph.number_of_nodes()
    #     epsilon = epsilons[np.argmin([abs(avg_degree - np.argwhere(diff_adj >= e).shape[0] / diff_adj.shape[0])
    #                                   for e in epsilons])]

    diff_edges = np.nonzero(diff_adj)
    diff_graph = dgl.graph(diff_edges)

    return graph, diff_graph, feat, label, train_idx, val_idx, test_idx

