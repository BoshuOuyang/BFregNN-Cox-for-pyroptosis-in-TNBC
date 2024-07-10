import networkx as nx
import numpy as np
from copy import deepcopy
from sklearn import preprocessing
import random  
import torch
import csv



def generate_network_architecture(graph_dicts, drug_gene_dicts, key1, key2, thres):

    def count_degree(graph_dicts,gene_list):
        degree_list = []
        return_graph_dicts = {}
        return_graph_dicts['edges'] = []
        for idx, g in enumerate(gene_list):
            degree_list.append(0)
            if g in graph_dicts:
                for end_nodes in graph_dicts[g]:
                    if end_nodes in gene_list and [idx,gene_list.index(end_nodes)] not in return_graph_dicts['edges']:
                        return_graph_dicts['edges'].append([idx,gene_list.index(end_nodes)]) 
                        degree_list[idx]+=1
        return degree_list


    def select_subgraph(graph_dicts,gene_list):
        gene_list = list(set(gene_list))
        gene_list.sort()
        return_graph_dicts = {}
        return_graph_dicts['nodes'] = []
        return_graph_dicts['nodes_name'] = []
        return_graph_dicts['edges'] = []
        for idx, g in enumerate(gene_list):
            return_graph_dicts['nodes'].append(idx)
            return_graph_dicts['nodes_name'].append(g)
            if g in graph_dicts:
                for end_nodes in graph_dicts[g]:
                    if end_nodes in gene_list and [idx,gene_list.index(end_nodes)] not in return_graph_dicts['edges']:
                        return_graph_dicts['edges'].append([idx,gene_list.index(end_nodes)]) 
                        
            return_graph_dicts['edges'].append([idx,idx])
        # return_graph_dicts['edges']=list(set(return_graph_dicts['edges']))
        return return_graph_dicts


    def trans_dicts2graph(graph_dicts):
        G = nx.Graph()
        for key in graph_dicts.keys():
            for k in graph_dicts[key]:
                G.add_edges_from([(key,k)])
        return G
        

    def shortest_path(G,source,target):
        try:
            num = nx.shortest_path_length(G, source, target)
        except: 
            return float('inf')
        return num
    

    def trans_layer(Gs, source_list, target_list):
        affin_graph = []
        removed_target = deepcopy(target_list)
        removed_source = []
        for idx_s, g_s in enumerate(source_list):
            marks_gene = {}
            min_num = float('inf')
            for idx, g_t in enumerate(target_list):
                num = shortest_path(Gs,g_s,g_t)
                if num not in marks_gene:
                    marks_gene[num] = []
                marks_gene[num].append(idx)
                if num > thres:
                    if g_t in removed_target:
                        removed_target.remove(g_t)
                if num < min_num:
                    min_num = num
            if min_num > thres:
                removed_source.append(g_s)
                continue
            for key in marks_gene.keys():
                # print(key,marks_gene[key])
                if key <= thres:
                    for g in marks_gene[key]:
                        affin_graph.append([idx_s,g])
        return removed_source, removed_target, np.array(affin_graph)
    

    def max_connect(basic_layer,second_layer,transfer_layer):
        G = nx.Graph()
        removed = []
        basic_lens = len(basic_layer['nodes'])
        for e in basic_layer['edges']:
            G.add_edge(e[0], e[1])
        for e in second_layer['edges']:
            G.add_edge(e[0] + basic_lens, e[1] + basic_lens)
        for e in transfer_layer:
            G.add_edge(e[0], e[1] + basic_lens)

        largest_graph = nx.connected_components(G)
        nodes = []
        for n in second_layer['nodes']:
            nodes.append(n + basic_lens)
        id_list = []
        for c in largest_graph:
            for n in c:
                if n in nodes:
                    id_list.append(1)
                else:
                    id_list.append(0)
        for idx, c in largest_graph:
            for n in c:
                if id_list[idx] == 0:
                    if n in basic_layer['nodes']:
                        removed.append(basic_layer['nodes_name'][n])
        return removed
    

    Gs = trans_dicts2graph(graph_dicts)

    focus_genes_list = []
    with open('../data/gene_pyroptosis_9.txt') as f:
        for line in f.readlines():
            focus_genes_list.append(line.split()[0])

    first_layer_gene = drug_gene_dicts[key1] + drug_gene_dicts[key2]
    first_layer_gene.sort()  
    removed_s, removed_t, transfer_layer = trans_layer(Gs, first_layer_gene, focus_genes_list)
    second_layer_graph = select_subgraph(graph_dicts, focus_genes_list)


    drug_list = drug_gene_dicts[key1] + drug_gene_dicts[key2]
    drug_list.sort()  

    degree_list = count_degree(graph_dicts,drug_list)
    drug_list = [d for idx, d in enumerate(drug_list) if not (d in removed_s and degree_list[idx]==0)]  


    basic_layer_graph = select_subgraph(graph_dicts, drug_list)
    first_layer_gene = basic_layer_graph['nodes_name']
    removed = max_connect(basic_layer_graph, second_layer_graph, transfer_layer)
    removed_s, removed_t, transfer_layer = trans_layer(Gs, first_layer_gene, focus_genes_list)
    removed_all = removed_s + removed


    drug_list = [d for idx, d in enumerate(drug_list) if not d in removed_all]  
    basic_layer_graph = select_subgraph(graph_dicts, drug_list)
    first_layer_gene = basic_layer_graph['nodes_name']
    removed_s, removed_t, transfer_layer = trans_layer(Gs, first_layer_gene, focus_genes_list)

    return basic_layer_graph, transfer_layer, second_layer_graph

def read_global_graph(thres_score):
    protein_id_dicts = {}

    with open('../data/9606.protein.info.v11.5.txt') as f:
        for idx, line in enumerate(f.readlines()):
            if idx == 0:
                continue
            line = line.split()
            protein_id_dicts[line[0]] = line[1]

    protein_graphs = {}
    protein_graphs['nodes_name'] = []
    protein_graphs['nodes'] = []
    protein_graphs['edges'] = []
    protein_graphs['edge_att'] = []

    graph_dicts = {}
    graph_dicts['nodes'] = {}
    with open('../data/9606.protein.links.full.v11.5.txt') as f:
        for idx, line in enumerate(f.readlines()):
            if idx == 0:
                continue
            line = line.split()
            source = protein_id_dicts[line[0]]
            target = protein_id_dicts[line[1]]
            score = float(line[-1])/1000
            if score >= thres_score:
                if source not in protein_graphs['nodes_name']:
                    protein_graphs['nodes_name'].append(source)
                    protein_graphs['nodes'].append(protein_graphs['nodes_name'].index(source))
                
                if target not in protein_graphs['nodes_name']:
                    protein_graphs['nodes_name'].append(target)
                    protein_graphs['nodes'].append(protein_graphs['nodes_name'].index(target))
                if source not in graph_dicts:
                    graph_dicts[source] = []
                graph_dicts[source].append(target)
                protein_graphs['edges'].append([source,target])
                protein_graphs['edge_att'].append(score)
    return graph_dicts

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)  
    torch.manual_seed(seed)  
    if torch.cuda.is_available():  
        torch.cuda.manual_seed_all(seed)  

    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False 


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