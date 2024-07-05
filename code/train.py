import numpy as np
from sklearn import preprocessing
import os
from torch.utils.data import Dataset
import csv
import torch
import torch.nn as nn
from model import BFRegNN_COX


def read_patient_info(gene_list):
    csv_reader = csv.reader(open("../data/tnbc2022_448s_gene_tpm_matrix.csv"))
    first_line = True
    X_index = {}
    for line in csv_reader:
        if first_line:
            temp_id_list = line[1:]
            first_line = False
        else:
            gene_name = line[0].strip('\n').strip(' ')
            i = gene_name.find('|')
            gene_name = gene_name[i+1:]
            if gene_name in gene_list:
                X_index[gene_name] = line[1:]

    X_all = []
    for gene_name in gene_list:
        X_all.append(X_index[gene_name])
    X_all=np.array(X_all)
    X_all = X_all.transpose()

    x_sample_id_list = []
    for sample_id in temp_id_list:
        pos = sample_id.find("_rep")
        if pos>0:
            sample_id = sample_id[0:pos]
        x_sample_id_list.append(sample_id)
    
    gene_index = gene_list

    csv_reader = csv.reader(open("../data/FUSCCTNBC_info.csv"))
    first_line = True
    y_tumor = {} # 1-tumor 0-normal
    y_subtypes = {} # 1-LAR 2-MES 3-BLIS 4-IM  0-Normal
    for line in csv_reader:
        if first_line:
            first_line = False
        else:
            sample_id = line[1].strip('\n').strip(' ')
            if line[2] == "tumor":
                y_tumor[sample_id] = 1
            elif line[2] == "normal":
                y_tumor[sample_id] = 0
            
            if line[3] == "LAR":
                y_subtypes[sample_id] = 1
            elif line[3] == "MES":
                y_subtypes[sample_id] = 2
            elif line[3] == "BLIS":
                y_subtypes[sample_id] = 3
            elif line[3] == "IM":
                y_subtypes[sample_id] = 4
            elif line[3] == "Normal":
                y_subtypes[sample_id] = 0

    first_line = True
    pos_needed = ["PATIENT_ID", "RFS_STATUS", "RFS_TIME_DAYS", "RFS_TIME_MONTHS"]
    pos_id_needed = []
    y_survival = {}
    with open("../data/fuscctnbc_clinical_patient.txt","r") as f:
        for line in f:
            line = line.strip(' ').strip('\n')
            items = line.split("\t")
            if first_line:
                i = 0
                for item in items:
                    if item in pos_needed:
                        pos_id_needed.append(i)
                    i = i + 1
                first_line = False
            else:
                s_id = items[pos_id_needed[0]]
                if s_id in y_subtypes:
                    if items[pos_id_needed[1]]=="0":
                        y_survival[s_id]=(False, items[pos_id_needed[2]])
                    if items[pos_id_needed[1]]=="1":
                        y_survival[s_id]=(True, items[pos_id_needed[2]])

    x_data = []
    index = []
    y_data_label = []
    y_data_time = []
    for i, sample_id in enumerate(x_sample_id_list):
        if y_subtypes[sample_id] != 0:
            x_data.append(X_all[i])
            index.append(sample_id)
            y_data_label.append(y_survival[sample_id][0])
            y_data_time.append(float(y_survival[sample_id][1]))
    
    x_data = np.array(x_data)
    y_data_label = np.expand_dims(np.array(y_data_label),axis=0)
    y_data_time = np.expand_dims(np.array(y_data_time),axis=0)
    y_data = np.concatenate((y_data_label,y_data_time),axis=0).T
    return x_data, y_data

def normalize_data(X, y):
    scaler = preprocessing.StandardScaler().fit(X)
    X_transformed = scaler.transform(X) 
    return X_transformed, y

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
    
    model = BFRegNN_COX(gene_num, gene_num2, 64, ori_gene, transfer_layer, ori_gene2, device, cox_weights_list).to(device)    

    return model


def train_model(model, optimizer, train_data, device):
    epoch = 500
    patience = 40
    patience_count = 0
    global_loss = 0
    global_con = 0

    for e in range(epoch):
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
