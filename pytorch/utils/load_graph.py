# Please set your dataset access path correctly

import torch
import dgl
import torch as th
import dgl.function as fn
from cpu_mem_usage import get_memory
import time
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator
from sklearn.preprocessing import StandardScaler
import os 
import scipy
import numpy as np
import json

def get_ogb_evaluator(dataset):
	"""
	Get evaluator from Open Graph Benchmark based on dataset
	"""
	evaluator = Evaluator(name=dataset)
	return lambda preds, labels: evaluator.eval({
		"y_true": labels.view(-1, 1),
		"y_pred": preds.view(-1, 1),
	})["acc"]

	
def prepare_data(g, n_classes, args, device):

	tmp = (g.in_degrees()==0) & (g.out_degrees()==0)
	isolated_nodes = torch.squeeze(torch.nonzero(tmp, as_tuple=False))
	g.remove_nodes(isolated_nodes)
	
	feats = g.ndata.pop('feat')      
	labels = g.ndata.pop('label')

	train_nid = torch.nonzero(g.ndata['train_mask'], as_tuple=True)[0]
	val_nid = torch.nonzero(g.ndata['val_mask'], as_tuple=True)[0]
	test_nid = torch.nonzero(~(g.ndata['train_mask'] | g.ndata['val_mask']), as_tuple=True)[0]
	print('success----------------------------------------')
	print(len(train_nid))
	print(len(val_nid))
	print(len(test_nid))
	print(f"# Nodes: {g.number_of_nodes()}\n"
			f"# Edges: {g.number_of_edges()}\n"
			f"# Train: {len(train_nid)}\n"
			f"# Val: {len(val_nid)}\n"
			f"# Test: {len(test_nid)}\n"
			f"# Classes: {n_classes}\n")
	
	data = g,  feats, labels, n_classes, train_nid, val_nid, test_nid
	return data

def load_amazon():
	prefix = '/share/home/wangyan/cherry/graph'
	adj_full = scipy.sparse.load_npz('{}/adj_full.npz'.format(prefix)).astype(bool)
	g = dgl.from_scipy(adj_full)
	num_nodes = g.num_nodes()

	adj_train = scipy.sparse.load_npz('{}/adj_train.npz'.format(prefix)).astype(bool)
	train_nid = np.array(list(set(adj_train.nonzero()[0])))

	role = json.load(open('{}/role.json'.format(prefix)))
	mask = np.zeros((num_nodes,), dtype=bool)
	train_mask = mask.copy()
	train_mask[role['tr']] = True
	val_mask = mask.copy()
	val_mask[role['va']] = True
	test_mask = mask.copy()
	test_mask[role['te']] = True

	feats = np.load('{}/feats.npy'.format(prefix))
	scaler = StandardScaler()
	scaler.fit(feats[train_nid])
	feats = scaler.transform(feats)

	class_map = json.load(open('{}/class_map.json'.format(prefix)))
	class_map = {int(k): v for k, v in class_map.items()}
	num_classes = len(list(class_map.values())[0])
	class_arr = np.zeros((num_nodes, num_classes))
	for k, v in class_map.items():
		class_arr[k] = v
	
	g.ndata['feat'] = torch.tensor(feats, dtype=torch.float)
	g.ndata['label'] = torch.tensor(class_arr, dtype=torch.float)
	g.ndata['train_mask'] = torch.tensor(train_mask, dtype=torch.bool)
	g.ndata['val_mask'] = torch.tensor(val_mask, dtype=torch.bool)
	g.ndata['test_mask'] = torch.tensor(test_mask, dtype=torch.bool)

	return g, num_classes


def load_karate():
	from dgl.data import KarateClubDataset

	# load Karate data
	# data = KarateClubDataset()
	# g = data[0]
	u=torch.tensor([0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 4, 1, 5, 3])
	v=torch.tensor([1, 0, 2, 3, 4, 1, 3, 4, 2, 4, 3, 5, 1, 6])
	g=dgl.graph((u, v),num_nodes=7)
	print('karate data')
	print(g.ndata)
	print(g.edata)

	
	ndata=[]
	for nid in range(7):
		ndata.append((th.ones(4)*nid).tolist())
	ddd = {'feat': th.tensor(ndata)}
	g.ndata['label']=torch.tensor([0,1,0,1,0,1,1])
	g.ndata['feat'] = ddd['feat']
	print(g)

	

	# print(data[0].ndata)
	# g.ndata['labels'] = g.ndata['label']
	train_nid = th.tensor(range(0,4))
	val_nid = th.tensor(range(4,6))
	test_nid = th.tensor(range(6, 7))

	train_mask = torch.zeros((g.number_of_nodes(),), dtype=torch.bool)
	train_mask[train_nid] = True
	val_mask = torch.zeros((g.number_of_nodes(),), dtype=torch.bool)
	val_mask[val_nid] = True
	test_mask = torch.zeros((g.number_of_nodes(),), dtype=torch.bool)
	test_mask[test_nid] = True
	g.ndata['train_mask'] = train_mask
	g.ndata['val_mask'] = val_mask
	g.ndata['test_mask'] = test_mask

	return g, 2


def load_pubmed():
	from dgl.data import PubmedGraphDataset
	# load Pubmed data
	data = PubmedGraphDataset()
	g = data[0]
	# num_class = g.num_of_class
	# g.ndata['features'] = g.ndata['feat']
	# g.ndata['labels'] = g.ndata['label']
	g = dgl.remove_self_loop(g)
	return g, data.num_classes
def load_cora():
	from dgl.data import CoraGraphDataset
	# load cora data
	data = CoraGraphDataset()
	g = data[0]
	# g = dgl.from_networkx(data.graph)
	# g = g.long()
	# g = g.int()
	# g.ndata['features'] = g.ndata['feat']
	# g.ndata['labels'] = g.ndata['label']
	g = dgl.remove_self_loop(g)
	
	
	return g, data.num_classes


def load_reddit():
	from dgl.data import RedditDataset
	# load reddit data
	data = RedditDataset(self_loop=True, raw_dir="/home/cherry/graph/")
	g = data[0]
	g = dgl.remove_self_loop(g)
	return g, data.num_classes

def load_ogb(name, args):
	data = DglNodePropPredDataset(name=name, root="/home/cherry/graph/")
	
	# data = DglNodePropPredDataset(name=name)
	splitted_idx = data.get_idx_split()
	graph, labels = data[0]

	graph = dgl.remove_self_loop(graph) 

	labels = labels[:, 0]
	graph.ndata['label'] = labels

	in_feats = graph.ndata['feat'].shape[1]
	num_labels = len(th.unique(labels[th.logical_not(th.isnan(labels))]))

	# Find the node IDs in the training, validation, and test set.
	train_nid, val_nid, test_nid = splitted_idx['train'], splitted_idx['valid'], splitted_idx['test']
	train_mask = th.zeros((graph.number_of_nodes(),), dtype=th.bool)
	train_mask[train_nid] = True
	val_mask = th.zeros((graph.number_of_nodes(),), dtype=th.bool)
	val_mask[val_nid] = True
	test_mask = th.zeros((graph.number_of_nodes(),), dtype=th.bool)
	test_mask[test_nid] = True
	graph.ndata['train_mask'] = train_mask
	graph.ndata['val_mask'] = val_mask
	graph.ndata['test_mask'] = test_mask
	
	return graph, num_labels

def inductive_split(g):
	"""Split the graph into training graph, validation graph, and test graph by training
	and validation masks.  Suitable for inductive models."""
	train_g = g.subgraph(g.ndata['train_mask'])
	val_g = g.subgraph(g.ndata['train_mask'] | g.ndata['val_mask'])
	test_g = g
	return train_g, val_g, test_g
