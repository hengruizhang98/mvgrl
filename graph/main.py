import argparse
import numpy as np
import torch as th
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import dgl
from dgl.data import GINDataset
from dgl.dataloading import GraphDataLoader
import warnings

from evaluate_embedding import logreg

warnings.filterwarnings('ignore')

from evaluate_embedding import logreg, linearsvc
from model import MVGRL, LogReg

parser = argparse.ArgumentParser(description='mvgrl')

parser.add_argument('--dataname', type=str, default='MUTAG', help='Name of dataset.')
parser.add_argument('--gpu', type=int, default=0, help='GPU index. Default: 0, using cuda:0.')
parser.add_argument('--epochs', type=int, default=20, help='Training epochs.')
parser.add_argument('--patience', type=int, default=20, help='Patient epochs to wait before early stopping.')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate of mvgrl.')
parser.add_argument('--wd', type=float, default=0., help='Weight decay of mvgrl.')

parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
parser.add_argument('--k', type=int, default=5, help='APPNP parameter')
parser.add_argument('--alpha', type=float, default=0.2, help='APPNP parameter')

parser.add_argument('--n_layers', type=int, default=4, help='Number of GNN layers')

parser.add_argument("--hid_dim", type=int, default=32, help='Hidden layer dimensionalities.')

args = parser.parse_args()

# check cuda
if args.gpu != -1 and th.cuda.is_available():
    args.device = 'cuda:{}'.format(args.gpu)
else:
    args.device = 'cpu'


def collate(samples):
    ''' collate function for building graph dataloader'''

    graphs, labels = map(list, zip(*samples))

    # generate batched graphs and labels
    batched_graph = dgl.batch(graphs)
    batched_labels = th.tensor(labels)

    n_graphs = len(graphs)
    graph_id = th.arange(n_graphs)
    graph_id = dgl.broadcast_nodes(batched_graph, graph_id)

    batched_graph.ndata['graph_id'] = graph_id

    return batched_graph, batched_labels

if __name__ == '__main__':

    dataset = 
    graphs, labeled_graphs, labels = map(list, zip(*dataset))

    # generate a full-graph with all examples for evaluation
    wholegraph = dgl.batch(graphs)
    wholegraph.ndata['attr'] = wholegraph.ndata['attr'].to(th.float32)

    # create dataloader for batch training
    dataloader = GraphDataLoader(dataset,
                                 batch_size=args.batch_size,
                                 collate_fn=collate,
                                 drop_last=False,
                                 shuffle=True)

    in_dim = wholegraph.ndata['attr'].shape[1]

    # Step 2: Create model =================================================================== #
    model = MVGRL(in_dim, args.hid_dim, args.k, args.alpha, args.n_layers)
    model = model.to(args.device)

    # Step 3: Create training components ===================================================== #
    optimizer = th.optim.Adam(model.parameters(), lr=args.lr)


    print('===== Before training ======')

    wholegraph = wholegraph.to(args.device)
    wholefeat = wholegraph.ndata['attr']

    embs = model.get_embedding(wholegraph, wholefeat)
    lbls = th.LongTensor(labels).cpu()
    acc_mean, acc_std = linearsvc(embs, lbls)
    print('accuracy_mean, {:.4f}'.format(acc_mean))

    best_svc = 0
    best_svc_epoch = 0

    # Step 4: training epochs =============================================================== #
    for epoch in range(args.epochs):
        loss_all = 0
        model.train()

        for graph, label in dataloader:
            graph = graph.to(args.device)
            feat = graph.ndata['attr']
            graph_id = graph.ndata['graph_id']

            n_graph = label.shape[0]

            optimizer.zero_grad()
            loss = model(graph, feat, graph_id)
            loss_all += loss.item()
            loss.backward()
            optimizer.step()

        print('Epoch {}, Loss {:.4f}'.format(epoch, loss_all))

        embs = model.get_embedding(wholegraph, wholefeat)
        acc_mean, acc_std = linearsvc(embs, lbls)
        print('accuracy_mean, {:.4f}'.format(acc_mean))

    # print('Training End')
    # print('best logreg {:4f} ,best svc {:4f}'.format(best_logreg, best_svc))





