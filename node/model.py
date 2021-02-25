import numpy as numpy
import torch as th
import torch.nn as nn

from dgl.nn.pytorch import GraphConv
from dgl.nn.pytorch.glob import AvgPooling


class Discriminator(nn.Module):
    def __init__(self, dim):
        super(Discriminator, self).__init__()
        self.fn = nn.Bilinear(dim, dim, 1)


    def forward(self, h1, h2, h3, h4, c1, c2):

        c_x1 = th.unsqueeze(c1, 1)
        c_x1 = c_x1.expand_as(h1).contiguous()
        c_x2 = th.unsqueeze(c2, 1)
        c_x2 = c_x2.expand_as(h2).contiguous()

        # positive
        sc_1 = th.squeeze(self.f_k(h2, c_x1), 2)
        sc_2 = th.squeeze(self.f_k(h1, c_x2), 2)

        # negative
        sc_3 = th.squeeze(self.f_k(h4, c_x1), 2)
        sc_4 = th.squeeze(self.f_k(h3, c_x2), 2)

        logits = th.cat((sc_1, sc_2, sc_3, sc_4), 1)

        return logits


class Model(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Model, self).__init__()
        self.encoder1 = GraphConv(in_dim, out_dim, bias = True, norm = 'none')
        self.encoder2 = GraphConv(in_dim, out_dim, bias = True, norm = 'none')

        self.act = nn.Sigmoid()
        self.pooling = AvgPooling()

    def forward(self, graph, dif_graph, feat):
        1

        h1 = self.encoder1(graph, feat)
        h2 = self.encoder2(dif_graph, feat)

        c1 = self.act(self.pooling(h1))
        c2 = self.act(self.pooling(h2))

        h3 = self.encoder1(shuf_feat, graph)
        h4 = self.encoder2(shuf_feat, dif_graph)

        return h1, h2, c1, c2, h3, h4


