import argparse
import numpy as np
import torch as th
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import dgl
import warnings

warnings.filterwarnings('ignore')

from process import process_dataset
from model import Model, LogReg


def argument():
    parser = argparse.ArgumentParser(description='mvgrl')

    parser.add_argument('--dataname', type=str, default='cora', help='Name of dataset.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index. Default: -1, using CPU.')
    parser.add_argument('--epochs', type=int, default=3000, help='Training epochs.')
    parser.add_argument('--early_stopping', type=int, default=200, help='Patient epochs to wait before early stopping.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')

    parser.add_argument('--sample_size', type=int, default=2000, help='Number of nodes within a subgraph')
    parser.add_argument('--batch_size', type=int, default=4, help='Number of subgraphs in a training batch')

    parser.add_argument('--wd', type=float, default=0.0, help='Weight decay, L2 reg.')

    parser.add_argument("--hid_dim", type=int, default=512, help='Hidden layer dimensionalities.')

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

    patience = 20

    graph, diff_graph, feat, label, train_idx, val_idx, test_idx, diff_weight = process_dataset(args.dataname)

    feat_dim = feat.shape[1]
    n_classes = np.unique(label).shape[0]

    lbl1 = th.ones(args.batch_size, args.sample_size * 2)
    lbl2 = th.zeros(args.batch_size, args.sample_size * 2)
    lbl = th.cat((lbl1, lbl2), 1)

    ''' shuffle features as negative samples '''
    model = Model(feat_dim, args.hid_dim)

    graph = graph.add_self_loop()
    diff_graph.edata['weight'] = th.tensor(diff_weight).float()

    model = model.to(args.device)
    lbl = lbl.to(args.device)

    b_xent = nn.BCEWithLogitsLoss()
    xent = nn.CrossEntropyLoss()

    optimizer = th.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    cnt_wait = 0
    best = 1e9
    best_t = 0

    for epoch in range(100):

        model.train()
        optimizer.zero_grad()
        idx = np.random.randint(0, feat.shape[0] - args.sample_size + 1, args.batch_size)

        bg, bd = [], []
        shuf_idx = np.random.permutation(args.sample_size)

        for i in idx:
            g = dgl.node_subgraph(graph, list(range(i, i + args.sample_size)))
            dg = dgl.node_subgraph(diff_graph, list(range(i, i + args.sample_size)))

            fts = feat[i: i + args.sample_size]
            shuf_fts = fts[shuf_idx, :]

            g.ndata['feat'] = fts
            g.ndata['shuf_feat'] = shuf_fts

            dg.ndata['feat'] = fts
            dg.ndata['shuf_feat'] = shuf_fts

            bg.append(g)
            bd.append(dg)

        bg = dgl.batch(bg)
        bd = dgl.batch(bd)

        bf = bg.ndata.pop('feat')
        shuf_feat = bg.ndata.pop('shuf_feat')
        weight = bd.edata.pop('weight')

        bg = bg.to(args.device)
        bd = bd.to(args.device)

        bf = bf.to(args.device)
        shuf_feat = shuf_feat.to(args.device)
        weight = weight.to(args.device)

        out = model(bg, bd, bf, shuf_feat, weight)
        loss = b_xent(out, lbl)
        loss.backward()
        optimizer.step()

        print('Epoch: {0}, Loss: {1:0.8f}'.format(epoch, loss.item()))

        if loss < best:
            best = loss
            best_t = epoch
            cnt_wait = 0
            th.save(model.state_dict(), 'model.pkl')
        else:
            cnt_wait += 1

        if cnt_wait == patience:
            if verbose:
                print('Early stopping!')
            break

    # if verbose:
    #     print('Loading {}th epoch'.format(best_t))
    model.load_state_dict(th.load('model.pkl'))

    feat = feat.to(args.device)
    graph = graph.to(args.device)
    diff_graph = diff_graph.to(args.device)
    train_idx = train_idx.to(args.device)
    val_idx = val_idx.to(args.device)
    test_idx = test_idx.to(args.device)

    embeds, _ = model.get_embedding(graph, diff_graph, feat, th.tensor(diff_weight).float().cuda())

    train_embs = embeds[train_idx]
    test_embs = embeds[test_idx]

    label = label.to(args.device)

    train_lbls = label[train_idx]
    test_lbls = label[test_idx]

    accs = []
    wd = 0.01 if args.dataname == 'citeseer' else 0.0

    for _ in range(10):
        log = LogReg(args.hid_dim, n_classes)
        opt = th.optim.Adam(log.parameters(), lr=1e-2, weight_decay=wd)
        log = log.to(args.device)

        for epoch in range(300):
            log.train()
            opt.zero_grad()

            logits = log(train_embs)
            loss = xent(logits, train_lbls)

            loss.backward()
            opt.step()

        logits = log(test_embs)
        preds = th.argmax(logits, dim=1)
        acc = th.sum(preds == test_lbls).float() / test_lbls.shape[0]
        accs.append(acc * 100)

    accs = th.stack(accs)
    print(accs.mean().item(), accs.std().item())


