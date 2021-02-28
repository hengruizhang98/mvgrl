import numpy as numpy
import torch as th
import torch.nn as nn

from dgl.nn.pytorch import GraphConv
from dgl.nn.pytorch.glob import AvgPooling

class LogReg(nn.Module):
    def __init__(self, hid_dim, n_classes):
        super(LogReg, self).__init__()

        self.fc = nn.Linear(hid_dim, n_classes)

    def forward(self, seq):
        ret = th.log_softmax(self.fc(seq), dim = -1)
        return ret


class Discriminator(nn.Module):
    def __init__(self, dim):
        super(Discriminator, self).__init__()
        self.fn = nn.Bilinear(dim, dim, 1)


    def forward(self, h1, h2, h3, h4, c1, c2):

        dim0 = c1.shape[0]
        dim2 = c1.shape[1]

        h1 = h1.view(dim0, -1, dim2)
        h2 = h2.view(dim0, -1, dim2)
        h3 = h3.view(dim0, -1, dim2)
        h4 = h4.view(dim0, -1, dim2)

        c_x1 = th.unsqueeze(c1, 1)
        c_x1 = c_x1.expand_as(h1).contiguous()
        c_x2 = th.unsqueeze(c2, 1)
        c_x2 = c_x2.expand_as(h2).contiguous()

        # positive
        sc_1 = th.squeeze(self.fn(h2, c_x1), 2)
        sc_2 = th.squeeze(self.fn(h1, c_x2), 2)

        # negative
        sc_3 = th.squeeze(self.fn(h4, c_x1), 2)
        sc_4 = th.squeeze(self.fn(h3, c_x2), 2)

        logits = th.cat((sc_1, sc_2, sc_3, sc_4), 1)

        return logits


class Model(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Model, self).__init__()
        self.encoder1 = GraphConv(in_dim, out_dim, bias = True, norm = 'none')
        self.encoder2 = GraphConv(in_dim, out_dim, bias = True, norm = 'none')

        self.act = nn.Sigmoid()
        self.pooling = AvgPooling()

        self.disc = Discriminator(out_dim)

    def get_embedding(self, graph, dif_graph, feat):
        h1 = self.encoder1(graph, feat)
        h2 = self.encoder2(dif_graph, feat)

        c = self.pooling(graph, h1)

        return (h1+h2).detach(), c.detach()

    def forward(self, graph, dif_graph, feat, shuf_feat):

        h1 = self.encoder1(graph, feat)
        h2 = self.encoder2(dif_graph, feat)

        c1 = self.act(self.pooling(graph, h1))
        c2 = self.act(self.pooling(dif_graph, h2))

        h3 = self.encoder1(graph, shuf_feat)
        h4 = self.encoder2(dif_graph, shuf_feat)

        out = self.disc(h1, h2, h3, h4, c1, c2)

        return out


