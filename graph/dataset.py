from dgl.data import GINDataset

import numpy as np
import networkx as nx
import torch
from scipy.linalg import fractional_matrix_power, inv
import scipy.sparse as sp
import dgl
import torch as th

def compute_ppr(graph: nx.Graph, alpha=0.2, self_loop=True):
    a = nx.convert_matrix.to_numpy_array(graph)
    if self_loop:
        a = a + np.eye(a.shape[0])  # A^ = A + I_n
    d = np.diag(np.sum(a, 1))  # D^ = Sigma A^_ii
    dinv = fractional_matrix_power(d, -0.5)  # D^(-1/2)
    at = np.matmul(np.matmul(dinv, a), dinv)  # A~ = D^(-1/2) x A^ x D^(-1/2)
    return alpha * inv((np.eye(a.shape[0]) - (1 - alpha) * at))  # a(I_n-(1-a)A~)^-1


def load_dataset(name):
    dataset = GINDataset(name, self_loop=False)
    graphs, labels = map(list, zip(*dataset))

    diff_graphs = []

    for graph in graphs:
        attr = graph.ndata['attr']
        nx_g = dgl.to_networkx(graph)
        diff_adj = compute_ppr(nx_g, 0.2)

        diff_edges = np.nonzero(diff_adj)
        diff_weight = diff_adj[diff_edges]
        diff_graph = dgl.graph(diff_edges)
        diff_graph.ndata['attr'] = attr
        diff_graph.edata['weight'] = th.tensor(diff_weight).float()
        diff_graphs.append(diff_graph)
        print(diff_graph)

    return zip(graphs, diff_graphs, labels)

if __name__ == '__main__':
    load_dataset('MUTAG')