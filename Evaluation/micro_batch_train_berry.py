import sys
sys.path.insert(0,'..')
sys.path.insert(0,'../pytorch/utils/')
sys.path.insert(0,'../pytorch/micro_batch_train/')
sys.path.insert(0,'../pytorch/models/')

import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import time
import argparse
import random
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from cherry_graph_partitioner import Graph_Partitioner, get_global_graph_edges_ids_block
from graphsage_model import GraphSAGE
from gcn_model_cherry import GCN
from gat_model_cherry import GAT
from load_graph import load_reddit, load_ogb, prepare_data, load_amazon, load_karate
from load_graph import load_ogbn_dataset
from memory_usage import see_memory_usage
from utils import Logger


def _memory_segment_key(args):
	"""根据模型类型返回内存回归分段键。"""
	if args.model == 'GAT':
		return 'gat'
	if args.model == 'SAGE' and args.aggre.lower() == 'lstm':
		return 'lstm'
	return 'sum'


def _extract_memory_features(blocks, args, nfeat_dim):
	"""从采样得到的 blocks 中提取内存估计特征向量与统计量。"""
	num_input_nodes = len(blocks[0].srcdata[dgl.NID])
	num_output_nodes = len(blocks[-1].dstdata[dgl.NID])
	edge_count_sum = sum(int(block.num_edges()) for block in blocks)
	hidden_nodes_sum = sum(int(len(block.srcdata[dgl.NID])) for block in blocks)

	vhat_f = float(num_input_nodes * nfeat_dim)
	ehat = float(edge_count_sum)
	hidden_feature = float(hidden_nodes_sum * max(int(args.num_hidden), 1))

	if args.model == 'GAT':
		t_k = float(args.num_heads * edge_count_sum)
	elif args.model == 'SAGE' and args.aggre.lower() == 'lstm':
		t_k = float(edge_count_sum * max(int(args.num_hidden), 1))
	else:
		t_k = float(edge_count_sum)

	r_k = float(max(num_input_nodes - num_output_nodes, 0) * nfeat_dim)
	x_k = [1.0, vhat_f, ehat, hidden_feature, t_k, r_k]
	stats = {
		'num_input_nodes': num_input_nodes,
		'num_output_nodes': num_output_nodes,
		'edge_count_sum': edge_count_sum,
		'hidden_nodes_sum': hidden_nodes_sum,
		't_k': t_k,
		'r_k': r_k,
	}
	return x_k, stats


def _load_beta_model(beta_path):
	"""从 JSON 文件加载内存回归系数 beta。"""
	path = Path(beta_path)
	if not path.exists():
		return {}
	try:
		with path.open('r', encoding='utf-8') as f:
			data = json.load(f)
		return data.get('beta', {})
	except Exception as e:
		print("WARNING: failed to load beta model from {}: {}".format(beta_path, e))
		return {}


def _save_beta_model(beta_path, beta_dict, sample_count):
	"""将拟合后的 beta 系数及元数据保存到 JSON 文件。"""
	path = Path(beta_path)
	path.parent.mkdir(parents=True, exist_ok=True)
	payload = {
		'version': 1,
		'updated_at': time.time(),
		'sample_count': sample_count,
		'beta': beta_dict,
	}
	with path.open('w', encoding='utf-8') as f:
		json.dump(payload, f, indent=2)


def _append_profile_sample(profile_path, sample):
	"""将单条内存画像样本追加写入 JSONL 文件。"""
	path = Path(profile_path)
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open('a', encoding='utf-8') as f:
		f.write(json.dumps(sample) + '\n')


def _load_profile_samples(profile_path):
	"""从 JSONL 文件读取内存画像样本列表。"""
	path = Path(profile_path)
	if not path.exists():
		return []
	samples = []
	with path.open('r', encoding='utf-8') as f:
		for line in f:
			line = line.strip()
			if not line:
				continue
			try:
				samples.append(json.loads(line))
			except json.JSONDecodeError:
				continue
	return samples


def _fit_beta_from_samples(samples):
	"""按分段对样本做最小二乘拟合，得到 beta 系数。"""
	segment_samples = {'sum': [], 'gat': [], 'lstm': []}
	for sample in samples:
		segment = sample.get('segment', 'sum')
		x = sample.get('x_k', [])
		y = sample.get('real_peak_gb', None)
		if segment in segment_samples and isinstance(x, list) and len(x) == 6 and isinstance(y, (int, float)):
			segment_samples[segment].append((x, float(y)))

	beta_dict = {}
	for segment, pairs in segment_samples.items():
		if len(pairs) < 2:
			continue
		X = np.array([p[0] for p in pairs], dtype=np.float64)
		y = np.array([p[1] for p in pairs], dtype=np.float64)
		beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
		beta_dict[segment] = beta.tolist()

	return beta_dict


def _estimate_with_beta(x_k, segment, beta_dict):
	"""使用分段 beta 对特征向量进行线性内存估计。"""
	beta = beta_dict.get(segment)
	if not beta or len(beta) != len(x_k):
		return None
	return float(sum(v * b for v, b in zip(x_k, beta)))


def _profile_real_peak_memory_gb(model, blocks, nfeats, labels, device, args):
	"""通过一次前向+反向实际测量该微批次的 CUDA 峰值显存（GB）。"""
	if not torch.cuda.is_available() or 'cuda' not in str(device):
		return None

	criterion = nn.BCEWithLogitsLoss() if args.dataset == 'amazon' else nn.CrossEntropyLoss()
	was_training = model.training
	model.train()
	model.zero_grad(set_to_none=True)

	torch.cuda.empty_cache()
	torch.cuda.reset_peak_memory_stats()
	torch.cuda.synchronize()

	batch_inputs, batch_labels = load_block_subtensor(nfeats, labels, blocks, device, args)
	blocks_dev = [block.int().to(device) for block in blocks]
	batch_pred = model(blocks_dev, batch_inputs)
	if args.dataset == 'ogbn-papers100M':
		loss = criterion(batch_pred, batch_labels.long())
	elif args.dataset == 'amazon':
		loss = criterion(batch_pred, batch_labels.float())
	else:
		loss = criterion(batch_pred, batch_labels)
	loss.backward()
	torch.cuda.synchronize()

	peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
	model.zero_grad(set_to_none=True)
	if not was_training:
		model.eval()
	return float(peak_gb)


def _estimate_peak_memory_gb(model, blocks, args, nfeat_dim, num_classes):
	"""用轻量解析模型估计单个微批次的峰值显存（GB）。"""
	x_k, stats = _extract_memory_features(blocks, args, nfeat_dim)
	segment = _memory_segment_key(args)

	beta_peak = None
	if getattr(args, 'memory_use_regression', False):
		beta_peak = _estimate_with_beta(x_k, segment, getattr(args, '_memory_beta_dict', {}))
	if beta_peak is not None and beta_peak > 0:
		return beta_peak

	param_count = sum(p.numel() for p in model.parameters())
	bytes_fp = 4
	bytes_idx = 8
	bytes_label = 8

	num_input_nodes = stats['num_input_nodes']
	num_output_nodes = stats['num_output_nodes']
	edge_count_sum = stats['edge_count_sum']
	hidden_nodes_sum = stats['hidden_nodes_sum']

	# M_stable parts
	m_param = param_count * bytes_fp
	m_input = num_input_nodes * nfeat_dim * bytes_fp
	m_label = num_output_nodes * bytes_label
	m_block = 3 * bytes_idx * edge_count_sum
	m_hidden = hidden_nodes_sum * max(int(args.num_hidden), 1) * bytes_fp
	# Adam moments, xi = 2 in the thesis section
	m_opt = 2 * param_count * bytes_fp
	m_stable = m_param + m_input + m_label + m_block + m_hidden + m_opt

	# M_agg segmented by model/aggregator
	if args.model == 'GAT':
		m_agg = args.mem_beta_head * args.num_heads * edge_count_sum * bytes_fp
	elif args.model == 'SAGE' and args.aggre.lower() == 'lstm':
		# Approximate sum_i (L_i * B_i) with total message count (edges)
		m_agg = args.mem_c_lstm * edge_count_sum * max(int(args.num_hidden), 1) * bytes_fp
	else:
		m_agg = args.mem_c_sum * edge_count_sum * bytes_fp

	# M_grad
	m_grad = param_count * bytes_fp

	peak_bytes = m_stable + max(m_agg, m_grad)
	return peak_bytes / (1024 ** 3)


def _build_micro_dataloader(g, batch_list, fanouts, num_workers):
	"""为单个微批次构建邻居采样 DataLoader。"""
	sampler = dgl.dataloading.MultiLayerNeighborSampler(fanouts)
	return dgl.dataloading.DataLoader(
		g,
		batch_list,
		sampler,
		batch_size=len(batch_list),
		shuffle=True,
		drop_last=False,
		num_workers=num_workers
	)


def _prepare_micro_batch_cpu(micro_idx, block_dataloader, nfeats, labels):
	"""在 CPU 侧准备一个微批次：采样 blocks 并抽取对应特征与标签。"""
	t_prepare = time.time()
	for input_nodes, _, blocks in block_dataloader:
		batch_inputs = nfeats[blocks[0].srcdata[dgl.NID]]
		batch_labels = labels[blocks[-1].dstdata[dgl.NID]]
		prepare_time = time.time() - t_prepare
		stats = {
			'micro_idx': micro_idx,
			'num_input_nids': len(input_nodes),
			'num_src_node': get_compute_num_nids(blocks),
			'num_out_node_FL': get_FL_output_num_nids(blocks),
			'prepare_time': prepare_time,
		}
		return {
			'blocks': blocks,
			'batch_inputs': batch_inputs,
			'batch_labels': batch_labels,
			'stats': stats,
		}

	# 若 DataLoader 为空，返回空对象，调用方应跳过。
	return None


def _move_micro_batch_to_device(micro_pack, device, dataset, non_blocking=False):
	"""将 CPU 侧微批次数据搬移到设备侧，并返回搬移耗时。"""
	t_move = time.time()
	blocks_dev = [block.int().to(device) for block in micro_pack['blocks']]
	batch_inputs = micro_pack['batch_inputs'].to(device, non_blocking=non_blocking)
	batch_labels = micro_pack['batch_labels'].to(device, non_blocking=non_blocking)
	if torch.cuda.is_available() and 'cuda' in str(device):
		torch.cuda.synchronize()
	move_time = time.time() - t_move

	if dataset == 'ogbn-papers100M':
		batch_labels = batch_labels.long()

	return blocks_dev, batch_inputs, batch_labels, move_time


def _memory_aware_partition(g, batched_output_nid_list, model, args, nfeat_dim, num_classes, nfeats, labels, device):
	"""评估各微批次显存峰值估计，并决定是否需要重新划分。"""
	fanouts = [int(fanout) for fanout in args.fan_out.split(',')]
	estimated_list = []
	for micro_idx, batch_list in enumerate(batched_output_nid_list):
		if len(batch_list) == 0:
			estimated_list.append(0.0)
			continue
		dataloader = _build_micro_dataloader(g, batch_list, fanouts, 0)
		for _, _, blocks in dataloader:
			x_k, _ = _extract_memory_features(blocks, args, nfeat_dim)
			segment = _memory_segment_key(args)
			est_gb = _estimate_peak_memory_gb(model, blocks, args, nfeat_dim, num_classes)
			estimated_list.append(est_gb)
			print("Memory estimate | Micro-batch-{} | {:.4f} GB".format(micro_idx, est_gb))

			if args.memory_profile_collect and args._memory_profile_collected < args.memory_profile_max_samples:
				real_peak_gb = _profile_real_peak_memory_gb(model, blocks, nfeats, labels, device, args)
				if real_peak_gb is not None:
					sample = {
						'timestamp': time.time(),
						'segment': segment,
						'x_k': x_k,
						'estimated_peak_gb': est_gb,
						'real_peak_gb': real_peak_gb,
					}
					_append_profile_sample(args.memory_profile_path, sample)
					args._memory_profile_collected += 1
					print("Memory profile sample collected ({}/{}), est={:.4f} GB, real={:.4f} GB".format(
						args._memory_profile_collected,
						args.memory_profile_max_samples,
						est_gb,
						real_peak_gb,
					))
			break

	max_estimated = max(estimated_list) if estimated_list else 0.0
	threshold = args.memory_budget_gb * args.memory_safety_factor
	print("Memory estimate summary | max: {:.4f} GB | budget: {:.4f} GB | safety-threshold: {:.4f} GB".format(
		max_estimated,
		args.memory_budget_gb,
		threshold,
	))
	return max_estimated <= threshold, max_estimated


def set_seed(args):
	"""设置 Python/NumPy/PyTorch/DGL 随机种子，提升结果可复现性。"""
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	if args.device >= 0:
		torch.cuda.manual_seed_all(args.seed)
		torch.cuda.manual_seed(args.seed)
		torch.backends.cudnn.enabled = False
		torch.backends.cudnn.deterministic = True
		dgl.seed(args.seed)
		dgl.random.seed(args.seed)


def compute_acc(pred, labels):
	"""根据预测结果与标签计算分类准确率。"""
	labels = labels.long()
	return (torch.argmax(pred, dim=1) == labels).float().sum() / len(pred)

def evaluate(model, g, nfeats, labels, train_nid, val_nid, test_nid, device, args):
	"""在全图上执行推理并返回 train/val/test 三个划分的准确率。"""
	nfeats=nfeats.to(device)
	g=g.to(device)
	# print('device ', device)
	model.eval()
	with torch.no_grad():
		# pred = model(g=g, x=nfeats)
		pred = model.inference(g, nfeats,  args, device)
	model.train()
	
	train_acc= compute_acc(pred[train_nid], labels[train_nid].to(pred.device))
	val_acc=compute_acc(pred[val_nid], labels[val_nid].to(pred.device))
	test_acc=compute_acc(pred[test_nid], labels[test_nid].to(pred.device))
	return (train_acc, val_acc, test_acc)


def load_block_subtensor(nfeat, labels, blocks, device,args):
	"""按当前 blocks 抽取输入特征和目标标签，并搬移到指定设备。"""

	# if args.GPUmem:
	# 	see_memory_usage("----------------------------------------before batch input features to device")
	batch_inputs = nfeat[blocks[0].srcdata[dgl.NID]].to(device)
	# if args.GPUmem:
	# 	see_memory_usage("----------------------------------------after batch input features to device")
	batch_labels = labels[blocks[-1].dstdata[dgl.NID]].to(device)
	# if args.GPUmem:
	# 	see_memory_usage("----------------------------------------after  batch labels to device")
	return batch_inputs, batch_labels

def get_compute_num_nids(blocks):
	"""统计一个微批次中所有层参与计算的源节点总数。"""
	res=0
	for b in blocks:
		res+=len(b.srcdata['_ID'])
	return res

	
def get_FL_output_num_nids(blocks):
	"""统计第一层输出节点数量。"""
	
	output_fl =len(blocks[0].dstdata['_ID'])
	return output_fl

def _partition_train_nodes_once(g, train_nid, args):
	"""执行一次训练节点划分，返回微批次节点列表及对应权重。"""
	max_neighbor = g.in_degrees(train_nid).max()
	print("max_neighbor: ", max_neighbor)
	
    # one partition
	sampler = dgl.dataloading.MultiLayerNeighborSampler([max_neighbor])
	full_batch_size = len(train_nid)
	args.num_workers = 0
	full_batch_dataloader = dgl.dataloading.DataLoader(
		g,
		train_nid,
		sampler,
		# device='cpu',
		batch_size=full_batch_size,
		shuffle=True,
		drop_last=False,
		num_workers=args.num_workers)
	
	if args.selection_method =='Metis':
		args.o_graph = dgl.node_subgraph(g, train_nid)

	batched_output_nid_list = []
	weights_list = []
	
	t1 = time.time()
	if args.selection_method == 'Metis' or args.selection_method == 'Cherry' or args.selection_method == 'Berry':
		for _,(src_full, dst_full, full_blocks) in enumerate(full_batch_dataloader):
			for layer_id, layer_block in enumerate(reversed(full_blocks)):
				block_eidx_global, block_edges_nids_global = get_global_graph_edges_ids_block(g, layer_block)
				layer_block.edata['_ID'] = block_eidx_global
				if layer_id == 0:
					my_graph_partitioner=Graph_Partitioner(layer_block, args)
					batched_output_nid_list, weights_list, p_len_list=my_graph_partitioner.init_graph_partition()
	elif args.selection_method == 'Range':
		micro_batch_size = len(train_nid) // args.num_batch
		start_index = 0
		for i in range(args.num_batch):
			end_index = min(start_index + micro_batch_size, len(train_nid))
			batched_output_nid_list.append(train_nid[start_index:end_index])
			weights_list.append((end_index - start_index)/len(train_nid))
			start_index = end_index
	elif args.selection_method == 'Random':
		train_nid_list = list(train_nid)
		random.shuffle(train_nid_list)
		micro_batch_size = len(train_nid) // args.num_batch
		start_index = 0
		for i in range(args.num_batch):
			end_index = min(start_index + micro_batch_size, len(train_nid))
			batched_output_nid_list.append(train_nid_list[start_index:end_index])
			weights_list.append((end_index - start_index)/len(train_nid))
			start_index = end_index
		
	return batched_output_nid_list, weights_list


def gen_micro_batch(g, train_nid, args, model, nfeat_dim, num_classes, nfeats, labels, device):
	"""基于划分策略生成微批次，并可按显存预算自适应增加批次数。"""
	base_num_batch = args.num_batch
	batch_step = 0
	t1 = time.time()

	if args.memory_use_regression and not hasattr(args, '_memory_beta_dict'):
		args._memory_beta_dict = _load_beta_model(args.memory_beta_path)
		if args._memory_beta_dict:
			print("Loaded beta model from {}".format(args.memory_beta_path))

	if not hasattr(args, '_memory_profile_collected'):
		args._memory_profile_collected = 0

	while True:
		print("Partition attempt {} | selection={} | num_batch={}".format(
			batch_step + 1,
			args.selection_method,
			args.num_batch,
		))
		batched_output_nid_list, weights_list = _partition_train_nodes_once(g, train_nid, args)

		if (not args.memory_aware_partition) or args.memory_budget_gb <= 0:
			break

		is_safe, max_estimated = _memory_aware_partition(
			g,
			batched_output_nid_list,
			model,
			args,
			nfeat_dim,
			num_classes,
			nfeats,
			labels,
			device,
		)
		if is_safe:
			print("Memory-aware partition converged with num_batch={} (max_estimated={:.4f} GB).".format(
				args.num_batch,
				max_estimated,
			))
			break

		if batch_step >= args.memory_max_partition_steps:
			print("WARNING: reach memory-aware max steps ({}), keep current num_batch={} with estimated peak {:.4f} GB.".format(
				args.memory_max_partition_steps,
				args.num_batch,
				max_estimated,
			))
			break

		args.num_batch += 1
		batch_step += 1
		print("Memory budget exceeded, increase num_batch to {} and repartition.".format(args.num_batch))

	print("one partition time: ", time.time() - t1)
	print("Weights List:", weights_list)
	print("Final num_batch after partition: {} (initial {}).".format(args.num_batch, base_num_batch))

	if args.memory_fit_beta:
		samples = _load_profile_samples(args.memory_profile_path)
		beta_dict = _fit_beta_from_samples(samples)
		if beta_dict:
			_save_beta_model(args.memory_beta_path, beta_dict, len(samples))
			args._memory_beta_dict = beta_dict
			print("Fitted beta model from {} samples, saved to {}".format(len(samples), args.memory_beta_path))
		else:
			print("WARNING: insufficient profiling samples for beta fitting.")
	
	return batched_output_nid_list, weights_list

def gen_model(args, in_feats, out_feats, device):
	"""根据参数选择并构建对应的 GNN 模型。"""
	if args.model == "SAGE":
		model = GraphSAGE(
			in_feats,
			args.num_hidden,
			out_feats,
			args.aggre,
			args.num_layers,
			F.relu,
			args.dropout
		).to(device)
	elif args.model == "GCN":
		model = GCN(
			in_feats,
			args.num_hidden,
			out_feats,
			args.num_layers,
			F.relu,
			args.dropout
		).to(device)
	elif args.model == "GAT":
		model = GAT(
			in_feats,
			args.num_hidden,
			out_feats,
			args.num_heads,
			args.num_layers,
			F.relu,
			args.dropout
		).to(device)
	
	return model

#### Entry point
def run(args, device, data):
	"""执行训练主流程：划分微批次、训练循环与评估统计。"""
	# Unpack data
	g, nfeats, labels, n_classes, train_nid, val_nid, test_nid = data
	in_feats = len(nfeats[0])
	print('in feats: ', in_feats)
	print('classes: ', n_classes)

	full_batch_size = len(train_nid)
	infer_batch_size = int(full_batch_size/args.num_batch) + (full_batch_size % args.num_batch>0)
	args.batch_size = infer_batch_size
	model = gen_model(args, in_feats, n_classes, device)
	# Micro-batch generate
	batched_output_nid_list, weights_list = gen_micro_batch(
		g,
		train_nid,
		args,
		model,
		in_feats,
		n_classes,
		nfeats,
		labels,
		device,
	)

	if args.memory_calibrate_only:
		print("Calibration-only mode enabled, skip training loop.")
		return None

	# Micro-batch dataloader
	fanouts = [int(fanout) for fanout in args.fan_out.split(',')]
	sampler = dgl.dataloading.MultiLayerNeighborSampler(fanouts)
	args.num_workers = 0
	batch_dataloaders = []
	for batch_list in batched_output_nid_list:
		dataloader = dgl.dataloading.DataLoader(
			g,
			batch_list,
			sampler,
			batch_size = len(batch_list),
			shuffle=True,
			drop_last=False,
			num_workers=args.num_workers
        )
		batch_dataloaders.append(dataloader)
	
					
	logger = Logger(args.num_runs, args)

	for run in range(args.num_runs):
		model.reset_parameters()
		criterion = nn.BCEWithLogitsLoss() if args.dataset == 'amazon' else nn.CrossEntropyLoss()
		optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
		for epoch in range(args.num_epochs):
			print("EPOCH BEGIN-----------------------------")
			total_t = time.time()

			num_src_node =0
			num_out_node_FL=0
			num_input_nids=0
			num_total_nids=0
			model.train()
			loss_sum=0
			pseudo_mini_loss = torch.tensor([], dtype=torch.long)

			load_block_time = []
			block_move_time = []
			model_time = []
			loss_time = []
			prefetch_wait_time = []
			opt_time = []

			use_async_prefetch = bool(args.async_prefetch)
			micro_count = len(batch_dataloaders)

			if use_async_prefetch:
				print("Async prefetch enabled | workers={} | non_blocking_copy={}".format(
					args.prefetch_workers,
					args.prefetch_non_blocking,
				))
				worker_num = max(int(args.prefetch_workers), 1)
				prefetch_pool = ThreadPoolExecutor(max_workers=worker_num)
				future_map = {}

				for micro_idx in range(micro_count):
					future_map[micro_idx] = prefetch_pool.submit(
						_prepare_micro_batch_cpu,
						micro_idx,
						batch_dataloaders[micro_idx],
						nfeats,
						labels,
					)

				for micro_idx in range(micro_count):
					t_wait = time.time()
					micro_pack = future_map[micro_idx].result()
					prefetch_wait_time.append(time.time() - t_wait)
					if micro_pack is None:
						continue

					load_block_time.append(micro_pack['stats']['prepare_time'])
					num_input_nids += micro_pack['stats']['num_input_nids']
					num_src_node += micro_pack['stats']['num_src_node']
					num_out_node_FL += micro_pack['stats']['num_out_node_FL']

					# torch.cuda.empty_cache()
					torch.cuda.reset_max_memory_allocated()
					torch.cuda.synchronize() # synchronized

					blocks, batch_inputs, batch_labels, move_t = _move_micro_batch_to_device(
						micro_pack,
						device,
						args.dataset,
						non_blocking=args.prefetch_non_blocking,
					)
					block_move_time.append(move_t)

					t4 = time.time()
					batch_pred = model(blocks, batch_inputs)
					torch.cuda.synchronize() # synchronized
					t5 = time.time()
					model_time.append(t5 - t4)

					pseudo_mini_loss = criterion(batch_pred, batch_labels)
					pseudo_mini_loss = pseudo_mini_loss * weights_list[micro_idx]
					pseudo_mini_loss.backward()

					loss_sum += pseudo_mini_loss

					torch.cuda.synchronize() # synchronized
					t2 = time.time()
					loss_time.append(t2 - t5)

					max_memory_allocated = torch.cuda.max_memory_allocated() / (1024 * 1024 * 1024)
					print("Micro-batch-", micro_idx, " max memory allocated: ", max_memory_allocated, " GB")

				prefetch_pool.shutdown(wait=True)
			else:
				for micro_idx, block_dataloader in enumerate(batch_dataloaders):
					# torch.cuda.empty_cache()
					torch.cuda.reset_max_memory_allocated()
					torch.cuda.synchronize() # synchronized
					for _, (_, _, blocks) in enumerate(block_dataloader):
						num_input_nids += len(blocks[0].srcdata[dgl.NID])
						num_src_node += get_compute_num_nids(blocks)
						num_out_node_FL += get_FL_output_num_nids(blocks)

						t1 = time.time()
						batch_inputs, batch_labels = load_block_subtensor(nfeats, labels, blocks, device,args)
						t3 = time.time()
						load_block_time.append(t3 - t1)
						blocks = [block.int().to(device) for block in blocks]
						torch.cuda.synchronize() # synchronized
						t4 = time.time()
						block_move_time.append(t4 - t3)
						batch_pred = model(blocks, batch_inputs)
						torch.cuda.synchronize() # synchronized
						t5 = time.time()
						model_time.append(t5 - t4)

						if args.dataset=='ogbn-papers100M':
							pseudo_mini_loss = criterion(batch_pred, batch_labels.long())
						elif args.dataset=='amazon':
							pseudo_mini_loss = criterion(batch_pred, batch_labels.float())
						else:
							pseudo_mini_loss = criterion(batch_pred, batch_labels)
						pseudo_mini_loss = pseudo_mini_loss * weights_list[micro_idx]
						pseudo_mini_loss.backward()

						loss_sum += pseudo_mini_loss

						torch.cuda.synchronize() # synchronized
						t2 = time.time()
						loss_time.append(t2 - t5)

						max_memory_allocated = torch.cuda.max_memory_allocated() / (1024 * 1024 * 1024)
						print("Micro-batch-", micro_idx, " max memory allocated: ", max_memory_allocated, " GB")
						break
			
			torch.cuda.synchronize() # synchronized
			opt_t = time.time()
			optimizer.step()
			optimizer.zero_grad()
			torch.cuda.synchronize() # synchronized
			opt_time.append(time.time() - opt_t)
			if args.GPUmem:
					see_memory_usage("-----------------------------------------after optimizer zero grad")
			if args.eval:
				
				args.batch_size = len(train_nid)//args.num_batch +1

				train_acc, val_acc, test_acc = evaluate(model, g, nfeats, labels, train_nid, val_nid, test_nid, device, args)

				if args.GPUmem:
					see_memory_usage("-----------------------------------------after evaluate")

				logger.add_result(run, (train_acc, val_acc, test_acc))
					
				print("Run {:02d} | Epoch {:05d} | Loss {:.4f} | Train {:.4f} | Val {:.4f} | Test {:.4f}".format(run, epoch, loss_sum.item(), train_acc, val_acc, test_acc))
			else:
				print(' Run '+str(run)+'| Epoch '+ str( epoch)+' |')

			print("TIME RECORD-----------------------------")
			if use_async_prefetch:
				print("prefetch_wait_time: ", sum(prefetch_wait_time))
			print("load_block_time: ", sum(load_block_time))
			print("block_move_time: ", sum(block_move_time))
			print("model_time: ", sum(model_time))
			print("loss_time: ", sum(loss_time))
			print("optimizer_time: ", sum(opt_time))
			print("total_time: ", time.time() - total_t)
			print("NODES RECORD----------------------------")
			print('Number of nodes for computation during this epoch: ', num_src_node)
			print('Number of input nodes during this epoch: ', num_input_nids)
			print('Number of first layer output nodes during this epoch: ', num_out_node_FL)

	
def count_parameters(model):
	"""打印模型参数总量及可训练/不可训练参数明细。"""
	pytorch_total_params = sum(torch.numel(p) for p in model.parameters())
	print('total model parameters size ', pytorch_total_params)
	print('trainable parameters')
	
	for name, param in model.named_parameters():
		if param.requires_grad:
			print (name + ', '+str(param.data.shape))
	print('-'*40)
	print('un-trainable parameters')
	for name, param in model.named_parameters():
		if not param.requires_grad:
			print (name, param.data.shape)

def main():
	"""解析命令行参数、加载数据并启动训练入口。"""
	tt = time.time()
	print("main start at this time " + str(tt))
	argparser = argparse.ArgumentParser("multi-gpu training")
	argparser.add_argument('--device', type=int, default=0)
	argparser.add_argument('--seed', type=int, default=1236)
	argparser.add_argument('--setseed', type=bool, default=True)
	argparser.add_argument('--GPUmem', type=bool, default=True)
	argparser.add_argument('--load-full-batch', type=bool, default=True)
	argparser.add_argument('--dataset', type=str, default='ogbn-arxiv')
	argparser.add_argument('--aggre', type=str, default='mean')
	argparser.add_argument('--selection-method', type=str, default='Cherry')
	argparser.add_argument('--num-batch', type=int, default=2)
	argparser.add_argument('--num-runs', type=int, default=1)
	argparser.add_argument('--num-epochs', type=int, default=1)
	argparser.add_argument('--num-hidden', type=int, default=256)
	argparser.add_argument('--num-layers', type=int, default=1)
	argparser.add_argument('--fan-out', type=str, default='10')
	argparser.add_argument('--lr', type=float, default=1e-2)
	argparser.add_argument('--dropout', type=float, default=0.5)
	argparser.add_argument("--weight-decay", type=float, default=5e-4)
	argparser.add_argument("--eval", action='store_true')
	argparser.add_argument('--num-workers', type=int, default=4)
	argparser.add_argument('--device-number', type=str, default='0')
	argparser.add_argument('--num-heads', type=int, default=4)
	argparser.add_argument('--model', type=str, default='SAGE')
	argparser.add_argument('--memory-aware-partition', action='store_true')
	argparser.add_argument('--memory-budget-gb', type=float, default=0.0)
	argparser.add_argument('--memory-safety-factor', type=float, default=1.0)
	argparser.add_argument('--memory-max-partition-steps', type=int, default=8)
	argparser.add_argument('--mem-c-sum', type=float, default=1.0)
	argparser.add_argument('--mem-c-lstm', type=float, default=18.0)
	argparser.add_argument('--mem-beta-head', type=float, default=1.0)
	argparser.add_argument('--memory-use-regression', action='store_true')
	argparser.add_argument('--memory-profile-collect', action='store_true')
	argparser.add_argument('--memory-profile-path', type=str, default='Evaluation/berry/memory_profile_samples.jsonl')
	argparser.add_argument('--memory-profile-max-samples', type=int, default=32)
	argparser.add_argument('--memory-fit-beta', action='store_true')
	argparser.add_argument('--memory-beta-path', type=str, default='Evaluation/berry/memory_beta.json')
	argparser.add_argument('--memory-calibrate-only', action='store_true')
	argparser.add_argument('--async-prefetch', action='store_true')
	argparser.add_argument('--prefetch-workers', type=int, default=2)
	argparser.add_argument('--prefetch-non-blocking', action='store_true')
	
	args = argparser.parse_args()

	os.environ["CUDA_VISIBLE_DEVICES"] = args.device_number

	if args.setseed:
		set_seed(args)
	device = "cpu"
	if args.GPUmem:
		see_memory_usage("-----------------------------------------before load data ")
	
	if args.dataset=='karate':
		g, n_classes = load_karate()
		print('#nodes:', g.number_of_nodes())
		print('#edges:', g.number_of_edges())
		print('#classes:', n_classes)
		device = "cuda:0"
		data=prepare_data(g, n_classes, args, device)
	elif args.dataset=='reddit':
		g, n_classes = load_reddit()
		device = "cuda:0"
		data=prepare_data(g, n_classes, args, device)
		print('#nodes:', g.number_of_nodes())
		print('#edges:', g.number_of_edges())
		print('#classes:', n_classes)
	elif args.dataset == 'ogbn-arxiv':
		data = load_ogbn_dataset(args.dataset,  args)
		device = "cuda:0"
	elif args.dataset=='ogbn-products':
		g, n_classes = load_ogb(args.dataset,args)
		print('#nodes:', g.number_of_nodes())
		print('#edges:', g.number_of_edges())
		print('#classes:', n_classes)
		device = "cuda:0"
		data=prepare_data(g, n_classes, args, device)
	elif args.dataset=='ogbn-papers100M':
		g, n_classes = load_ogb(args.dataset,args)
		print('#nodes:', g.number_of_nodes())
		print('#edges:', g.number_of_edges())
		print('#classes:', n_classes)
		device = "cuda:0"
		data=prepare_data(g, n_classes, args, device)
	elif args.dataset=='amazon':
		g, n_classes = load_amazon()
		print('#nodes:', g.number_of_nodes())
		print('#edges:', g.number_of_edges())
		print('#classes:', n_classes)
		device = "cuda:0"
		data=prepare_data(g, n_classes, args, device)
	else:
		raise Exception('unknown dataset')
		
	
	best_test = run(args, device, data)
	

if __name__=='__main__':
	main()

 