import pickle
import argparse
import torch
import os
import random  
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from train import build_bfregNN_model, GeneDataset, normalize_data, train_model, read_patient_info
from copy import deepcopy
import networkx as nx


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--score', type=float, default=0.6, help='Edge threshold for ppi network')
    parser.add_argument('--drug1', type=str, default="mitoxantrone", help='First drug')
    parser.add_argument('--drug2', type=str, default="gambogic acid", help='Second drug')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-1, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--seed', type=int, default=2222, help='Random seed for reproducibility')
    args = parser.parse_known_args()[0]
    return args

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

args = parse()
set_seed(args.seed)

if os.path.exists('../data/ppi_global_graph_'+str(args.score)+'.pkl'):
    with open('../data/ppi_global_graph_'+str(args.score)+'.pkl','rb') as f:
        global_graph = pickle.load(f)
else:
    global_graph = read_global_graph(args.score)
    with open('../data/ppi_global_graph_'+str(args.score)+'.pkl','wb') as f:
        pickle.dump(global_graph, f)


drug_dicts = {}
with open('../data/9_drug_targets_1.0_revised.tsv') as f:
    for line in f.readlines():
        line = line.split('\n')[0].split('\t')
        drug_dicts[line[0]]=line[1:]


basic_layer, transfer_layer, second_layer = generate_network_architecture(global_graph, drug_dicts, args.drug1, args.drug2, 2)
transfer_layer = np.array(transfer_layer).T
basic_layer_adj = np.array(basic_layer['edges']).T
second_layer_adj = np.array(second_layer['edges']).T


X, y = read_patient_info(basic_layer['nodes_name'])
X, y = normalize_data(X, y)
train_data_dataset = GeneDataset(X, y)
train_data = DataLoader(train_data_dataset, batch_size=args.batch_size)

cox_weights = {'CASP1':0.041459,'BCL2':-0.030420,'CTSD':-0.002452,'CASP5':-0.850369,'TRADD':-0.170639,'NFKB2':0.028377,'CASP9':0.19231,'TNF':-0.126771,'GSDMD':0.029954}
cox_weights_list = []
for g in second_layer['nodes_name']:
    cox_weights_list.append(cox_weights[g])
cox_weights_list = np.array(cox_weights_list)
cox_weights_list = np.expand_dims(cox_weights_list, axis=1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_bfregNN_model(X.shape[1], cox_weights_list.shape[0], basic_layer_adj, second_layer_adj, transfer_layer, device, cox_weights_list)
optimizer = optim.Adam(model.parameters(), lr = args.learning_rate, weight_decay = args.weight_decay)
loss, con_loss = train_model(model, train_data, optimizer, args, device)

with open('drug_combi_result.txt','a') as f:
    f.write(args.drug1 + '\t' + args.drug2 + '\t' + str(args.score) + '\t' + str(loss) + '\t' + str(con_loss) + '\n')