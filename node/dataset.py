from scipy.linalg import fractional_matrix_power, inv
import scipy.sparse as sp
import networkx as nx
import numpy as np
import os
import dgl
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
import torch as th

def compute_ppr(graph: nx.Graph, alpha=0.2, self_loop=True):
    a = nx.convert_matrix.to_numpy_array(graph)
    if self_loop:
        a = a + np.eye(a.shape[0])                                # A^ = A + I_n
    d = np.diag(np.sum(a, 1))                                     # D^ = Sigma A^_ii
    dinv = fractional_matrix_power(d, -0.5)                       # D^(-1/2)
    at = np.matmul(np.matmul(dinv, a), dinv)                      # A~ = D^(-1/2) x A^ x D^(-1/2)
    return alpha * inv((np.eye(a.shape[0]) - (1 - alpha) * at))   # a(I_n-(1-a)A~)^-1


def process_dataset(name, threshold):
    if name == 'cora':
        dataset = CoraGraphDataset()
    elif name == 'citeseer':
        dataset = CiteseerGraphDataset()
    elif name == 'pubmed':
        dataset = PubmedGraphDataset()

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
    diff_adj[diff_adj < threshold] = 0

    diff_edges = np.nonzero(diff_adj)
    diff_weight = diff_adj[diff_edges]
    diff_graph = dgl.graph(diff_edges)

    return graph, diff_graph, feat, label, train_idx, val_idx, test_idx, diff_weight

