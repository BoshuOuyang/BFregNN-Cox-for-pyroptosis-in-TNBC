import pickle
import argparse
import torch
import os
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from train import GeneDataset, build_bfregNN_model, train_model
from utils import generate_network_architecture, read_global_graph, set_seed, read_patient_info, normalize_data

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--score', type=float, default=0.6, help='Edge threshold for ppi network')
    parser.add_argument('--drug1', type=str, default="mitoxantrone", help='First drug')
    parser.add_argument('--drug2', type=str, default="gambogic acid", help='Second drug')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-1, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--seed', type=int, default=2222, help='Random seed for reproducibility')
    args = parser.parse_known_args()[0]
    return args


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

print(args.drug1, args.drug2, con_loss)

with open('drug_combi_result.txt','a') as f:
    f.write(args.drug1 + '\t' + args.drug2 + '\t' + str(args.score) + '\t' + str(loss) + '\t' + str(con_loss) + '\n')