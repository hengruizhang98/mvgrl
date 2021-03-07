import argparse
import numpy as np
import torch as th
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import dgl
import warnings

warnings.filterwarnings('ignore')

from full_process import process_dataset
from full_model import MVGRL, LogReg

parser = argparse.ArgumentParser(description='mvgrl')

parser.add_argument('--dataname', type=str, default='cora', help='Name of dataset.')
parser.add_argument('--gpu', type=int, default=0, help='GPU index. Default: 0, using cuda:0.')
parser.add_argument('--epochs', type=int, default=500, help='Training epochs.')
parser.add_argument('--early_stopping', type=int, default=200, help='Patient epochs to wait before early stopping.')
parser.add_argument('--lr1', type=float, default=0.001, help='Learning rate of mvgrl.')
parser.add_argument('--lr2', type=float, default=0.01, help='Learning rate of linear evaluator.')
parser.add_argument('--wd1', type=float, default=0., help='Weight decay of mvgrl.')
parser.add_argument('--wd2', type=float, default=0., help='Weight decay of linear evaluator.')

parser.add_argument('--full', action='store_true', default=True, help='Full-graph training')

parser.add_argument('--sample_size', type=int, default=2000, help='Number of nodes within a subgraph')
parser.add_argument('--batch_size', type=int, default=4, help='Number of subgraphs in a training batch')

parser.add_argument("--hid_dim", type=int, default=512, help='Hidden layer dimensionalities.')

args = parser.parse_args()

# check cuda
if args.gpu != -1 and th.cuda.is_available():
    args.device = 'cuda:{}'.format(args.gpu)
else:
    args.device = 'cpu'


def main(args):
    print(args)
    # Step 1: Prepare graph data and retrieve train/validation/test index ============================= #
    # Load from DGL dataset
    graph, feat, label, train_idx, val_idx, test_idx = process_dataset(args.dataname)

    n_feat = feat.shape[1]
    n_classes = np.unique(label).shape[0]

    lbl1 = th.ones(1, graph.number_of_nodes() * 2)
    lbl2 = th.zeros(1, graph.number_of_nodes() * 2)
    lbl = th.cat((lbl1, lbl2), 1)

    k = 20
    alpha = 0.2

    model = MVGRL(n_feat, args.hid_dim, k, alpha)

    graph = graph.add_self_loop()

    # move tensor to device
    model = model.to(args.device)
    feat = feat.to(args.device)
    model = model.to(args.device)
    label = label.to(args.device)
    lbl = lbl.to(args.device)
    graph = graph.to(args.device)

    b_xent = nn.BCEWithLogitsLoss()
    xent = nn.CrossEntropyLoss()

    optimizer = th.optim.Adam(model.parameters(), lr=args.lr1, weight_decay=args.wd1)

    N_NODE = graph.number_of_nodes()

    cnt_wait = 0
    best = float('inf')
    patience = 50

    for epoch in range(args.epochs):
        optimizer.zero_grad()

        shuf_idx = np.random.permutation(graph.number_of_nodes())
        shuf_feat = feat[shuf_idx, :].to(args.device)

        out = model(graph, feat, shuf_feat)
        loss = b_xent(out, lbl)
        loss.backward()
        optimizer.step()

        print('Epoch: {0}, Loss: {1:0.4f}'.format(epoch, loss.item()))

        if loss < best:
            best = loss
            cnt_wait = 0
            th.save(model.state_dict(), 'model.pkl')
        else:
            cnt_wait += 1

        if cnt_wait == patience:
            break

    model.load_state_dict(th.load('model.pkl'))

    embeds, _ = model.get_embedding(graph, feat)

    train_embs = embeds[train_idx]
    test_embs = embeds[test_idx]

    train_lbls = label[train_idx]
    test_lbls = label[test_idx]

    for _ in range(1):
        log = LogReg(args.hid_dim, n_classes)
        opt = th.optim.Adam(log.parameters(), lr=args.lr2, weight_decay=args.wd2)
        log = log.to(args.device)

        for epoch in range(400):
            log.train()
            opt.zero_grad()

            logits = log(train_embs)
            loss = xent(logits, train_lbls)

            loss.backward()
            opt.step()

            log.eval()
            logits = log(test_embs)
            preds = th.argmax(logits, dim=1)
            acc = th.sum(preds == test_lbls).float() / test_lbls.shape[0]
            print('epoch: {}, loss:{:.4f}, test_acc: {:.4f}'.format(epoch, loss.item(), acc))


if __name__ == '__main__':
    main(args)



