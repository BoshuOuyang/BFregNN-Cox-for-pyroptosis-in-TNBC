import torch
from torch import nn
import math
import numpy as np
from torch.nn import functional as F
from module import IntraLayer, InterLayer



class BFRegNN(nn.Module):
    def __init__(self, in_dim, in_dim2, n_hid, basic_layer, transfer_layer, second_layer, device):
        super().__init__()

        self.graph1 = basic_layer.to_dense()
        self.basic_graph = IntraLayer(in_dim, 1, 4, 4, "cat", device)

        self.transfer_graph = transfer_layer.to_dense()
        self.inter_layer = InterLayer(in_dim, in_dim2, device)

        self.graph2 = second_layer.to_dense()
        self.second_graph = IntraLayer(in_dim2, 4, 4, 4, "cat", device)

        self.cox_aff = cox_affine(in_dim2, device)


    def forward(self, x):
        x = x.unsqueeze(-1)

        x = self.basic_graph(x, self.graph1)
        x = self.inter_layer(x, self.transfer_graph)
        x = self.second_graph(x, self.graph2)
        x = self.cox_aff(x)
        
        return x
    
        
class BFRegNN_COX(nn.Module):
    def __init__(self, in_dim, in_dim2, n_hid, graphs1, transfer_layer, second_layer, device, cox_weights_list):
        super().__init__() 
        self.bfregNN = BFRegNN(in_dim, in_dim2, n_hid, graphs1, transfer_layer, second_layer, device)
        self.neg_module = cox_module(in_dim2, cox_weights_list, device)

    def forward(self, x, event, time):
        x = self.bfregNN(x)
        loss = self.neg_module(x, event, time)
        self.concordance = self.neg_module.concordance

        return loss


class cox_affine(nn.Module):
    def __init__(self, in_dim2, device):
        super().__init__()
        self.gene_num = in_dim2
        self.aff = nn.Linear(4,1)
    def forward(self,x):
        n_samples = x.shape[0]
        x.requires_grad_()
        x = x.reshape(n_samples, self.gene_num, -1)
        x = self.aff(x)
        x = x.squeeze()
        return x
        
        
        
class cox_module(nn.Module):
    def __init__(self, in_dim2, cox_weights_list, device):
        super().__init__()
        self.gene_num = in_dim2
        self.device = device

        self.W = torch.tensor(cox_weights_list, requires_grad=False).float().to(device)
    
    def forward(self, x, event, time, alpha=0, beta=0):
        
        _, o = torch.sort(-time, dim=0, stable=True)

        my_event = event[o]
        x = x[o,:]
        my_time = time[o]

        loss = 0
        xw = torch.matmul(x, self.W)
        loss, risksets, diff, pred = self.neg_par_log_likelihood(xw, my_time, my_event)
        loss = loss.mean()
        self.concordance = self.c_index(xw, my_time, my_event)
        return loss.unsqueeze(-1),(risksets, diff, pred, my_time, my_event)

    def neg_par_log_likelihood(self,pred, ytime, yevent):

        n_observed = yevent.sum(0)+1e-6
        ytime_indicator = self.R_set(ytime)
        
        risk_set_sum = ytime_indicator.mm(torch.exp(pred))  # hazard ratio
        diff = pred - torch.log(risk_set_sum)

        yevent = yevent.unsqueeze(-1)
        sum_diff_in_observed = torch.transpose(diff, 0, 1).mm(yevent)
        cost = ( -(sum_diff_in_observed / n_observed)).reshape((-1,))
        return(cost,risk_set_sum,diff,pred)
    
    def c_index(self,pred, ytime, yevent):

        n_sample = len(ytime)
        ytime_indicator = self.R_set(ytime)
        ytime_matrix = ytime_indicator - torch.diag(torch.diag(ytime_indicator))
        
        censor_idx = (yevent == 0).nonzero()
        zeros = torch.zeros(n_sample).to(self.device)
        ytime_matrix[censor_idx, :] = zeros

        pred_matrix = torch.zeros_like(ytime_matrix)

        pred_diffs = pred - pred.T
        pred_matrix[pred_diffs > 0] = 1  
        pred_matrix[pred_diffs == 0] = 0.5  


        concord_matrix = pred_matrix.mul(ytime_matrix)
        concord = torch.sum(concord_matrix)
        epsilon = torch.sum(ytime_matrix) + 1e-6
        concordance_index = torch.div(concord, epsilon)

        return concordance_index
    
    def R_set(self,x):

        n_sample = x.shape[0]
        matrix_ones = torch.ones(n_sample, n_sample).to(self.device)  
        indicator_matrix = torch.tril(matrix_ones).to(self.device)  

        return(indicator_matrix)

 