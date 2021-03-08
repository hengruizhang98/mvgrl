import torch as th
import torch.nn as nn

from dgl.nn.pytorch import GraphConv, APPNPConv
from dgl.nn.pytorch.glob import AvgPooling

class LogReg(nn.Module):
    def __init__(self, hid_dim, n_classes):
        super(LogReg, self).__init__()

        self.fc = nn.Linear(hid_dim, n_classes)

    def forward(self, seq):
        ret = self.fc(seq)
        return ret


class Discriminator(nn.Module):
    def __init__(self, dim):
        super(Discriminator, self).__init__()
        self.fn = nn.Bilinear(dim, dim, 1)

    def forward(self, h1, h2, h3, h4, c1, c2):

        c_x1 = c1.expand_as(h1).contiguous()
        c_x2 = c2.expand_as(h2).contiguous()

        # positive
        sc_1 = self.fn(h2, c_x1).squeeze(1)
        sc_2 = self.fn(h1, c_x2).squeeze(1)

        # negative
        sc_3 = self.fn(h4, c_x1).squeeze(1)
        sc_4 = self.fn(h3, c_x2).squeeze(1)

        logits = th.cat((sc_1, sc_2, sc_3, sc_4))

        return logits


class MVGRL(nn.Module):


    def __init__(self, in_dim, out_dim, k, alpha):
        super(MVGRL, self).__init__()

        self.encoder1 = GraphConv(in_dim, out_dim, norm='both', bias=True)
        self.encoder2 = APPNPConv(k, alpha)
        self.lin = nn.Linear(in_dim, out_dim, bias=True)
        self.pooling = AvgPooling()

        self.disc = Discriminator(out_dim)

        self.act1 = nn.PReLU()
        self.act2 = nn.Sigmoid()

    def get_embedding(self, graph, feat):
        h1 = self.encoder1(graph, feat)
        h2 = self.lin(self.encoder2(graph, feat))

        return (h1 + h2).detach()

    def forward(self, graph, feat, shuf_feat):
        h1 = self.act1(self.encoder1(graph, feat))
        h2 = self.act1(self.lin(self.encoder2(graph, feat)))

        h3 = self.act1(self.encoder1(graph, shuf_feat))
        h4 = self.act1(self.lin(self.encoder2(graph, shuf_feat)))

        c1 = self.act2(self.pooling(graph, h1))
        c2 = self.act2(self.pooling(graph, h2))

        out = self.disc(h1, h2, h3, h4, c1, c2)

        return out


