import argparse
import numpy as np
import torch as th
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import dgl
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset

import random
import warnings
warnings.filterwarnings('ignore')

from process import process_dataset
from model import Model

def argument():
    parser = argparse.ArgumentParser(description='GRAND')

    # data source params
    parser.add_argument('--dataname', type=str, default='cora', help='Name of dataset.')
    # cuda params
    parser.add_argument('--gpu', type=int, default=-1, help='GPU index. Default: -1, using CPU.')
    # training params
    parser.add_argument('--epochs', type=int, default=200, help='Training epochs.')
    parser.add_argument('--early_stopping', type=int, default=200, help='Patient epochs to wait before early stopping.')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='L2 reg.')
    # model params
    parser.add_argument("--hid_dim", type=int, default=32, help='Hidden layer dimensionalities.')
    parser.add_argument('--dropnode_rate', type=float, default=0.5,
                        help='Dropnode rate (1 - keep probability).')
    parser.add_argument('--input_droprate', type=float, default=0.5,
                        help='dropout rate of input layer')
    parser.add_argument('--hidden_droprate', type=float, default=0.5,
                        help='dropout rate of hidden layer')
    parser.add_argument('--order', type=int, default=8, help='Propagation step')
    parser.add_argument('--sample', type=int, default=4, help='Sampling times of dropnode')
    parser.add_argument('--tem', type=float, default=0.5, help='Sharpening temperature')
    parser.add_argument('--lam', type=float, default=1., help='Coefficient of consistency regularization')
    parser.add_argument('--use_bn', action='store_true', default=False, help='Using Batch Normalization')

    args = parser.parse_args()

    # check cuda
    if args.gpu != -1 and th.cuda.is_available():
        args.device = 'cuda:{}'.format(args.gpu)
    else:
        args.device = 'cpu'

    return args


if __name__ == '__main__':

    # Step 1: Prepare graph data and retrieve train/validation/test index ============================= #
    # Load from DGL dataset
    args = argument()
    print(args)

    graph, diff_graph, feat, label, train_idx, val_idx, test_idx = process_dataset(args.dataname)

    epochs = 3000
    patience = 20
    lr = 0.001
    l2 = 0.0
    hid_dim = 512
    sparse = False

    feat_dim = feat.shape[1]
    class_dim = np.unique(label).shape[0]

    sample_size = 2000
    batch_size = 4

    ''' shuffle featrues as negative samples '''

    model = Model(feat_dim, hid_dim)

    for epoch in range(epochs):

        idx = np.random.randint(0, feat.shape[0] - sample_size + 1, batch_size)
        ba, bd, bf = [], [], []

        for i in idx:
            ba.append(dgl.node_subgraph(graph, list(range(i, i+ sample_size))))




