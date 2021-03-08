import argparse
import numpy as np
import torch as th
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import dgl
import warnings

import random

warnings.filterwarnings('ignore')

from dataset import process_dataset
from model import MVGRL, LogReg

parser = argparse.ArgumentParser(description='mvgrl')

parser.add_argument('--dataname', type=str, default='cora', help='Name of dataset.')
parser.add_argument('--gpu', type=int, default=0, help='GPU index. Default: 0, using cuda:0.')
parser.add_argument('--epochs', type=int, default=500, help='Training epochs.')
parser.add_argument('--early_stopping', type=int, default=200, help='Patient epochs to wait before early stopping.')
parser.add_argument('--lr1', type=float, default=0.001, help='Learning rate of mvgrl.')
parser.add_argument('--lr2', type=float, default=0.01, help='Learning rate of linear evaluator.')
parser.add_argument('--wd1', type=float, default=0., help='Weight decay of mvgrl.')
parser.add_argument('--wd2', type=float, default=0., help='Weight decay of linear evaluator.')

parser.add_argument('--sample_size', type=int, default=2000, help='Number of nodes within a subgraph')
parser.add_argument('--k', type=int, default=20, help='Number of iterations of APPNP')
parser.add_argument('--alpha', type=float, default=0.2, help='Teleport probability of APPNP')

parser.add_argument("--hid_dim", type=int, default=512, help='Hidden layer dim.')

args = parser.parse_args()

# check cuda
if args.gpu != -1 and th.cuda.is_available():
    args.device = 'cuda:{}'.format(args.gpu)
else:
    args.device = 'cpu'


def train(args, model, optimizer, graph, feat):

    lbl1 = th.ones(args.sample_size * 2)
    lbl2 = th.zeros(args.sample_size * 2)
    lbl = th.cat((lbl1, lbl2))

    lbl = lbl.to(args.device)

    model = model.to(args.device)
    lbl = lbl.to(args.device)

    node_idx_list = list(range(graph.number_of_nodes()))

    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(args.epochs):
        model.train()

        optimizer.zero_grad()

        node_idx = random.sample(node_idx_list, args.sample_size)
        g = dgl.node_subgraph(graph, node_idx)
        fts = feat[node_idx]

        shuf_idx = np.random.permutation(args.sample_size)
        shuf_fts = fts[shuf_idx, :]

        g = g.to(args.device)
        fts = fts.to(args.device)
        shuf_fts = shuf_fts.to(args.device)

        out = model(g, fts, shuf_fts)
        loss = loss_fn(out, lbl)
        loss.backward()
        optimizer.step()

        print('Epoch: {0}, Loss: {1:0.4f}'.format(epoch, loss.item()))

        if loss < best:
            best = loss
            cnt_wait = 0
            th.save(model.state_dict(), 'model.pkl')
        else:
            cnt_wait += 1

        if cnt_wait == args.patience:
            break

def evaluate_embedding(args, train_embs, train_labels, test_embs, test_labels):

    model = LogReg(args.hid_dim, args.n_classes)
    opt = th.optim.Adam(model.parameters(), lr=args.lr2, weight_decay=args.wd2)

    model = model.to(args.device)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(300):
        model.train()
        opt.zero_grad()

        logits = model(train_embs)
        loss = loss_fn(logits, train_labels)
        loss.backward()
        opt.step()

    model.eval()
    logits = model(test_embs)
    preds = th.argmax(logits, dim=1)
    acc = th.sum(preds == test_labels).float() / test_labels.shape[0]

    return acc

def main(args):
    print(args)

    # Step 1: Prepare graph data and retrieve train/validation/test index ============================= #
    # Load from DGL dataset
    graph, feat, label, train_idx, val_idx, test_idx = process_dataset(args.dataname)

    n_feat = feat.shape[1]
    args.n_classes = np.unique(label).shape[0]

    model = MVGRL(n_feat, args.hid_dim, args.k, args.alpha)

    graph = graph.add_self_loop()
    optimizer = th.optim.Adam(model.parameters(), lr=args.lr1, weight_decay=args.wd1)

    train(args, model, optimizer, graph, feat)

    model.load_state_dict(th.load('model.pkl'))
    embeds = model.get_embedding(graph, feat)

    train_embs = embeds[train_idx]
    test_embs = embeds[test_idx]

    train_labels = label[train_idx]
    test_labels = label[test_idx]

    accs = []

    for _ in range(50):
        acc = evaluate_embedding(args, train_embs, train_labels, test_embs, test_labels)
        accs.append(acc*100)

    accs = th.stack(accs)
    print(accs.mean().item(), accs.std().item())


if __name__ == '__main__':
    main(args)



