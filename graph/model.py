import numpy as numpy
import torch as th
import torch.nn as nn

from dgl.nn.pytorch import GraphConv, APPNPConv
from dgl.nn.pytorch.glob import SumPooling

from utils import local_global_loss_

class LogReg(nn.Module):
    def __init__(self, hid_dim, n_classes):
        super(LogReg, self).__init__()

        self.fc = nn.Linear(hid_dim, n_classes)

    def forward(self, seq):
        ret = th.log_softmax(self.fc(seq), dim=-1)
        return ret


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MLP, self).__init__()
        self.fcs = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.PReLU(),
            nn.Linear(out_dim, out_dim),
            nn.PReLU(),
            nn.Linear(out_dim, out_dim),
            nn.PReLU()
        )
        self.linear_shortcut = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.fcs(x) + self.linear_shortcut(x)

class PPNP(nn.Module):
    def __init__(self, in_dim, out_dim, k, alpha):
        super(PPNP, self).__init__()

        self.conv = APPNPConv(k, alpha)
        self.lin = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.PReLU(),
        )

        self.pool = SumPooling()

    def forward(self, graph, feat):
        conv_feat = self.conv(graph, feat)
        out = self.lin(conv_feat)

        global_out = self.pool(graph, out)

        return out, global_out


class GCN(nn.Module):
    def __init__(self, in_dim, out_dim, num_layers):
        super(GCN, self).__init__()

        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        self.layers.append(GraphConv(in_dim, out_dim, bias=True, norm='both'))

        self.act_fn = nn.PReLU()
        self.pooling = SumPooling()

        for _ in range(num_layers - 1):
            self.layers.append(GraphConv(out_dim, out_dim, bias=True, norm='both'))

    def forward(self, graph, feat):
        h = self.act_fn(self.layers[0](graph, feat))
        hg = self.pooling(graph, h)

        for idx in range(self.num_layers - 1):
            h = self.act_fn(self.layers[idx + 1](graph, h))
            hg = th.cat((hg, self.pooling(graph, h)), -1)

        return h, hg


class MVGRL(nn.Module):
    '''
    Use APPNP to approximate PPNP
    '''

    def __init__(self, in_dim, out_dim, k, alpha, num_layers):
        super(MVGRL, self).__init__()
        self.local_mlp = MLP(out_dim, out_dim)
        self.global_mlp = MLP(num_layers * out_dim, out_dim)
        self.encoder1 = GCN(in_dim, out_dim, num_layers)
        self.ppnp = PPNP(in_dim, out_dim, k, alpha)
        self.encoder2 = GCN(out_dim, out_dim, num_layers-1)


    def get_embedding(self, graph, feat):
        local_v1, global_v1 = self.encoder1(graph, feat)
        out, out_global = self.ppnp(graph, feat)

        local_v2, global_v2 = self.encoder2(graph, out)

        global_v2 = th.cat((out_global, global_v2), dim = -1)

        global_v1 = self.global_mlp(global_v1)
        global_v2 = self.global_mlp(global_v2)

        return (global_v1 + global_v2).detach()

    def forward(self, graph, feat, graph_id):
        local_v1, global_v1 = self.encoder1(graph, feat)
        out, out_global = self.ppnp(graph, feat)

        local_v2, global_v2 = self.encoder2(graph, out)

        global_v2 = th.cat((out_global, global_v2), dim = -1)

        local_v1 = self.local_mlp(local_v1)
        local_v2 = self.local_mlp(local_v2)


        global_v1 = self.global_mlp(global_v1)
        global_v2 = self.global_mlp(global_v2)

        loss1 = local_global_loss_(local_v1, global_v2, graph_id)
        loss2 = local_global_loss_(local_v2, global_v1, graph_id)

        loss = loss1 + loss2

        return loss





