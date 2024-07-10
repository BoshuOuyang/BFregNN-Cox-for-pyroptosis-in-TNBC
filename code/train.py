import numpy as np
import os
from torch.utils.data import Dataset
import torch
import torch.nn as nn
from model import BFRegNN_COX


class GeneDataset(Dataset):
    def __init__(self, data, label):
        super(GeneDataset).__init__()
        self.x = data
        self.y = label
    def __getitem__(self, index):
        sample_x = self.x[index,:]
        sample_y = self.y[index,:]
        return [sample_x, sample_y]
    def __len__(self):
        return self.x.shape[0]


def build_bfregNN_model(gene_num, gene_num2, gene_adj, gene_adj2, transfer_layer, device, cox_weights_list):

    v1 = torch.ones(gene_adj.shape[1], device=device)  
    ori_gene = torch.sparse_coo_tensor(gene_adj, v1, size=(gene_num, gene_num))
    
    v2 = torch.ones(gene_adj2.shape[1], device=device) 
    ori_gene2 = torch.sparse_coo_tensor(gene_adj2, v2, size=(gene_num2, gene_num2))

    v3 = torch.ones(transfer_layer.shape[1], device=device) 
    transfer_layer = torch.sparse_coo_tensor(transfer_layer, v3, size=(gene_num,gene_num2)).to_dense()
    
    model = BFRegNN_COX(gene_num, gene_num2, 64, ori_gene, transfer_layer, ori_gene2, cox_weights_list).to(device)    

    return model


def train_model(model, train_data, optimizer, args, device):
    patience = 500
    patience_count = 0
    global_loss = 0
    global_con = 0

    for e in range(args.epochs):
        train_loss = 0
        concordance = 0
        for d in train_data:
            optimizer.zero_grad()
            data = d[0].float().to(device)
            label = d[1].float().to(device)
            event_label = label[:,0]
            time_label = label[:,1]

            loss, _ = model(data, event_label, time_label)
            loss = torch.mean(loss)
            loss.backward()
            optimizer.step()

            train_loss += loss
            concordance += model.concordance

        train_loss /= len(train_data)
        concordance /= len(train_data)

        print(e, concordance.item(), train_loss.item())

        if concordance > global_con:
            global_con = concordance
            global_loss = train_loss
            patience_count = 0
        else:
            patience_count += 1
            if patience_count == patience:
                break

    return global_loss.item(), global_con.item()
