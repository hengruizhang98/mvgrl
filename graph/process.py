from utils import preprocess_features, normalize_adj
from sklearn.preprocessing import MinMaxScaler
from utils import compute_ppr
import scipy.sparse as sp
import numpy as np
import dgl
from dgl.data import GINDataset
import torch as th


def process_dataset(name):
    dataset = GINDataset(name, self_loop = False)
    graphs, labels = map(list, zip(*dataset))

    return graphs, labels
