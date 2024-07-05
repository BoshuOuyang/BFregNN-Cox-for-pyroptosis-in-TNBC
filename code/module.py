import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GCN(nn.Module):
    def __init__(self, in_dim,  out_dim, device):
        super(GCN, self).__init__()

        self.W = nn.Parameter(torch.randn(in_dim, out_dim)).to(device)
        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))
        self.b = nn.Parameter(torch.randn(out_dim)).to(device)
        nn.init.uniform_(self.b)


    def forward(self, x, adj):
        support = torch.matmul(x, self.W)
        output = torch.matmul(adj, support)
        if self.b is not None:
            output = output + self.b

        return output
    

class IntraLayer(nn.Module):
    def __init__(self, n_nodes, in_dim, hid_dim, out_dim, mode, device):
        super(IntraLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.mode = mode
        self.n_nodes = n_nodes

        self.W_Q = nn.Linear(in_dim, out_dim)
        self.W_K = nn.Linear(in_dim, out_dim)        

        self.gcn = GCN(in_dim, hid_dim, device)
        self.batch_norm = nn.BatchNorm1d(n_nodes)
        self.cat_mlp = nn.Linear(in_dim + out_dim, out_dim)
        

    def forward(self, x, adj):
        Q = self.W_Q(x)  # Query
        K = self.W_K(x)  # Key

        attention_scores = torch.matmul(Q, K.transpose(-2, -1))
        attention_scores = F.sigmoid(attention_scores)

        adj_inv = (~adj.bool()).float()
        new_adj = attention_scores * (adj + 1e-5 * adj_inv)

        out = F.elu(self.gcn(x, new_adj))

        if self.mode == 'cat':
            out = self.cat_mlp(torch.cat([out, x], dim = -1))
            out = self.batch_norm(out)
            out = F.elu(out)
        else:
            print("mode error")

        return out


class InterLayer(nn.Module):
    def __init__(self, in_features, out_features, device):
        super(InterLayer, self).__init__()

        self.W = nn.Parameter(torch.randn(in_features, out_features)).to(device)
        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))

        self.b = nn.Parameter(torch.randn(out_features)).to(device)
        nn.init.uniform_(self.b)

        self.batch_norm = nn.BatchNorm1d(out_features)

    def forward(self, x, adj):

        adj_inv = (~adj.bool()).float()  
        w = self.W * (adj + 1e-2 * adj_inv)  
        w = w.transpose(1, 0)  
          
        x = x.transpose(1, 2)  
        x = F.linear(x, w, self.b)  
        x = x.transpose(1, 2)  
          
        x = self.batch_norm(x)  
        x = F.elu(x)

        return x


