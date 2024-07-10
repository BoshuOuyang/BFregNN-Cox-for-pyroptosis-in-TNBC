import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GCN(nn.Module):
    def __init__(self, in_dim,  out_dim):
        super(GCN, self).__init__()

        self.W = nn.Parameter(torch.randn(in_dim, out_dim))
        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))



    def forward(self, x, adj):
        support = torch.matmul(x, self.W)
        output = torch.matmul(adj, support)

        return output


class SGC(nn.Module):
    def __init__(self, in_dim,  out_dim):
        super(SGC, self).__init__()
        if in_dim != out_dim:
            print("SGC dim error")


    def forward(self, x, adj):
        output = torch.matmul(adj, x)

        return output


class IntraLayer(nn.Module):
    def __init__(self, n_nodes, in_dim, hid_dim, out_dim, mode, gnn_mode):
        super(IntraLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        if gnn_mode == "GCN":
            self.gnn = GCN(in_dim, hid_dim)
        elif gnn_mode == "SGC":
            self.gnn = SGC(in_dim, hid_dim)


    def forward(self, x, adj):

        out = F.elu(self.gnn(x, adj))

        return out


class InterLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(InterLayer, self).__init__()

        self.W = nn.Parameter(torch.randn(in_features, out_features))
        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))

        self.batch_norm = nn.BatchNorm1d(out_features)

    def forward(self, x, adj):

        w = self.W * adj 
        w = w.transpose(1, 0)  
          
        x = x.transpose(1, 2)  
        x = F.linear(x, w)
        x = x.transpose(1, 2)  

        x = self.batch_norm(x)  
        x = F.elu(x)

        return x


