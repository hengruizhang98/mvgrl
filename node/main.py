import argparse
import numpy as np
import torch as th
import torch.nn as nn

import warnings

warnings.filterwarnings('ignore')

from dataset import process_dataset
from model import MVGRL, LogReg

parser = argparse.ArgumentParser(description='mvgrl')

parser.add_argument('--dataname', type=str, default='cora', help='Name of dataset.')
parser.add_argument('--gpu', type=int, default=0, help='GPU index. Default: 0, using cuda:0.')
parser.add_argument('--epochs', type=int, default=500, help='Training epochs.')
parser.add_argument('--patience', type=int, default=20, help='Patient epochs to wait before early stopping.')
parser.add_argument('--lr1', type=float, default=0.001, help='Learning rate of mvgrl.')
parser.add_argument('--lr2', type=float, default=0.01, help='Learning rate of linear evaluator.')
parser.add_argument('--wd1', type=float, default=0., help='Weight decay of mvgrl.')
parser.add_argument('--wd2', type=float, default=0., help='Weight decay of linear evaluator.')

parser.add_argument('--threshold', type=float, default=0.001, help='edge mask threshold.')
parser.add_argument("--hid_dim", type=int, default=512, help='Hidden layer dim.')

args = parser.parse_args()

# check cuda
if args.gpu != -1 and th.cuda.is_available():
    args.device = 'cuda:{}'.format(args.gpu)
else:
    args.device = 'cpu'


def train(args, model, optimizer, graph, diff_graph, feat, edge_weight):
    n_node = graph.number_of_nodes()
    lbl1 = th.ones(n_node * 2)
    lbl2 = th.zeros(n_node * 2)
    lbl = th.cat((lbl1, lbl2))

    model = model.to(args.device)
    lbl = lbl.to(args.device)

    loss_fn = nn.BCEWithLogitsLoss()

    graph = graph.to(args.device)
    diff_graph = diff_graph.to(args.device)
    feat = feat.to(args.device)
    edge_weight = edge_weight.to(args.device)

    best = float('inf')
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()

        shuf_idx = np.random.permutation(n_node)
        shuf_feat = feat[shuf_idx, :]
        shuf_feat = shuf_feat.to(args.device)

        out = model(graph, diff_graph, feat, shuf_feat, edge_weight)
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

    graph, diff_graph, feat, label, train_idx, val_idx, test_idx, edge_weight = process_dataset(args.dataname,
                                                                                                args.threshold)
    n_feat = feat.shape[1]
    n_classes = np.unique(label).shape[0]

    graph = graph.to(args.device)
    diff_graph = diff_graph.to(args.device)
    feat = feat.to(args.device)
    edge_weight = edge_weight.to(args.device)
    edge_weight = th.tensor(edge_weight).float().to(args.device)

    train_idx = train_idx.to(args.device)
    val_idx = val_idx.to(args.device)
    test_idx = test_idx.to(args.device)

    n_node = graph.number_of_nodes()
    lbl1 = th.ones(n_node * 2)
    lbl2 = th.zeros(n_node * 2)
    lbl = th.cat((lbl1, lbl2))

    model = MVGRL(n_feat, args.hid_dim)
    model = model.to(args.device)

    lbl = lbl.to(args.device)

    optimizer = th.optim.Adam(model.parameters(), lr=args.lr1, weight_decay=args.wd1)

    loss_fn = nn.BCEWithLogitsLoss()

    best = float('inf')
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()

        shuf_idx = np.random.permutation(n_node)
        shuf_feat = feat[shuf_idx, :]
        shuf_feat = shuf_feat.to(args.device)

        out = model(graph, diff_graph, feat, shuf_feat, edge_weight)
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

    model.load_state_dict(th.load('model.pkl'))
    embeds = model.get_embedding(graph, diff_graph, feat, edge_weight)

    train_embs = embeds[train_idx]
    test_embs = embeds[test_idx]

    train_labels = label[train_idx]
    test_labels = label[test_idx]

    accs = []

    for _ in range(5):
        acc = evaluate_embedding(args, train_embs, train_labels, test_embs, test_labels)
        accs.append(acc * 100)

    accs = th.stack(accs)
    print(accs.mean().item(), accs.std().item())


if __name__ == '__main__':
    print(args)

    # Step 1: Prepare graph data and retrieve train/validation/test index ============================= #
    # Load from DGL dataset

    graph, diff_graph, feat, label, train_idx, val_idx, test_idx, edge_weight = process_dataset(args.dataname,
                                                                                                args.threshold)
    n_feat = feat.shape[1]
    n_classes = np.unique(label).shape[0]

    graph = graph.to(args.device)
    diff_graph = diff_graph.to(args.device)
    feat = feat.to(args.device)
    edge_weight = th.tensor(edge_weight).float().to(args.device)

    train_idx = train_idx.to(args.device)
    val_idx = val_idx.to(args.device)
    test_idx = test_idx.to(args.device)

    n_node = graph.number_of_nodes()
    lbl1 = th.ones(n_node * 2)
    lbl2 = th.zeros(n_node * 2)
    lbl = th.cat((lbl1, lbl2))

    model = MVGRL(n_feat, args.hid_dim)
    model = model.to(args.device)

    lbl = lbl.to(args.device)

    optimizer = th.optim.Adam(model.parameters(), lr=args.lr1, weight_decay=args.wd1)

    loss_fn = nn.BCEWithLogitsLoss()

    print(edge_weight)
    print(edge_weight.sum())

    best = float('inf')
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()

        shuf_idx = np.random.permutation(n_node)
        shuf_feat = feat[shuf_idx, :]
        shuf_feat = shuf_feat.to(args.device)

        out = model(graph, diff_graph, feat, shuf_feat, edge_weight)
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

    model.load_state_dict(th.load('model.pkl'))
    embeds = model.get_embedding(graph, diff_graph, feat, edge_weight)

    train_embs = embeds[train_idx]
    test_embs = embeds[test_idx]

    label = label.to(args.device)
    train_labels = label[train_idx]
    test_labels = label[test_idx]
    accs = []

    print(embeds)
    print(args.hid_dim, n_classes)
    for _ in range(5):
        model = LogReg(args.hid_dim, n_classes)
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
        accs.append(acc * 100)

    accs = th.stack(accs)
    print(accs.mean().item(), accs.std().item())



