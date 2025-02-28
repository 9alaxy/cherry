import dgl
import torch
import time
from my_utils import *
import numpy as np


def get_global_graph_edges_ids_block(raw_graph, block):
	
	edges=block.edges(order='eid', form='all')
	edge_src_local = edges[0]
	edge_dst_local = edges[1]
	induced_src = block.srcdata[dgl.NID]
	induced_dst = block.dstdata[dgl.NID]
		
	raw_src, raw_dst=induced_src[edge_src_local], induced_dst[edge_dst_local]
	
	# in homo graph: raw_graph 
	global_graph_eids_raw = raw_graph.edge_ids(raw_src, raw_dst)
	# https://docs.dgl.ai/generated/dgl.DGLGraph.edge_ids.html?highlight=graph%20edge_ids#dgl.DGLGraph.edge_ids
	# https://docs.dgl.ai/en/0.4.x/generated/dgl.DGLGraph.edge_ids.html#dgl.DGLGraph.edge_ids

	return global_graph_eids_raw, (raw_src, raw_dst)


class Graph_Partitioner:
	def __init__(self, layer_block, args):
		self.layer_block=layer_block
		self.local=False
		self.output_nids=layer_block.dstdata['_ID']
		self.local_output_nids=[]
		self.local_src_nids=[]
		self.src_nids_list= layer_block.srcdata['_ID'].tolist()
		self.full_src_len=len(layer_block.srcdata['_ID'])
		self.global_batched_seeds_list=[]
		self.local_batched_seeds_list=[]
		self.weights_list=[]
		self.num_batch=args.num_batch
		self.selection_method=args.selection_method
		self.ideal_partition_size=0

		self.partition_len_list=[]

		self.args=args

	def remove_non_output_nodes(self):
		import copy
		local_src=copy.deepcopy(self.local_src_nids)
		mask_array = np.full(len(local_src),True, dtype=np.bool_)
		mask_array[self.local_output_nids] = False
		from itertools import compress
		to_remove=list(compress(local_src, mask_array)) # time complexity O(n)
		return to_remove


	def get_src(self, seeds):
		in_ids=list(self.layer_block.in_edges(seeds))[0].tolist()
		src= list(set(in_ids+seeds))
		return src
	

	def simple_gen_K_batches_seeds_list(self):
		
		if self.selection_method=="Metis":
			o_graph = self.args.o_graph
			partition = dgl.metis_partition(g=o_graph,k=self.args.num_batch)
			res=[]
			for pid in partition:
				nids = partition[pid].ndata[dgl.NID].tolist()
				res.append(sorted(nids))
			if set(sum(res,[]))!=set(self.local_output_nids):
				print('--------pure    check:     the difference of graph partition res and self.local_output_nids')
			self.local_batched_seeds_list=res

		elif self.selection_method == "Cherry" or  self.selection_method == "vanilla":
			print('Out-degree Partition start----................................')
			t1 = time.time()
			
			# u -> src_node
			# v -> dst_node
			u, v =self.layer_block.edges()[0], self.layer_block.edges()[1] # local edges
			g = dgl.graph((u,v))

			# compute out degrees
			out_degrees = g.out_degrees(g.nodes())
			# Set weight as the src_nodes' out degrees
			g.edata['w'] = out_degrees[u]

			print("Out-degree Compute Time: ", time.time() - t1)

			t2 = time.time()
			to_remove = self.remove_non_output_nodes()
			if len(to_remove) > 0:
				g.remove_nodes(torch.tensor(to_remove))
			g_no_diag = dgl.remove_self_loop(g)
			print("Graph Check Time: ", time.time() - t2)

			print("--------Partition Graph Info")
			print("nodes: ", g.number_of_nodes())
			print("edges: ", g.number_of_edges())

			t3 = time.time()
			partition = dgl.metis_partition(g=g_no_diag,k=self.args.num_batch)
			print("Metis Time: ", time.time() - t3)
			print("Out-degree Partition Time: ", time.time() - t1)

			
			res=[]
			for pid in partition:
				nids = partition[pid].ndata[dgl.NID].tolist()
				res.append(sorted(nids))
				
			print('Partition end ----................................')
			self.local_batched_seeds_list=res
		return

	def get_src_len(self,seeds):
		in_ids=list(self.layer_block.in_edges(seeds))[0].tolist()
		src_len= len(list(set(in_ids+seeds)))
		return src_len



	def get_partition_src_len_list(self):
		partition_src_len_list=[]
		for seeds_nids in self.local_batched_seeds_list:
			partition_src_len_list.append(self.get_src_len(seeds_nids))
		
		self.partition_src_len_list=partition_src_len_list
		return partition_src_len_list

	def get_weight_list(self):
		weight_list=[]
		full_len = len(self.output_nids)
		for micro_idx, micro_batch_list in enumerate(self.local_batched_seeds_list):
			micro_batch_len = len(micro_batch_list)
			print("Micro-batch-", micro_idx, " train node: ", micro_batch_len)
			weight_list.append(micro_batch_len/full_len)
		return weight_list


	def graph_partition(self):
		
		self.ideal_partition_size = (self.full_src_len/self.num_batch)
		
		self.simple_gen_K_batches_seeds_list()

		# weight_list = get_weight_list(self.local_batched_seeds_list)
		src_len_list = self.get_partition_src_len_list()
		
		self.weights_list = self.get_weight_list()
		self.partition_len_list = src_len_list

		return self.local_batched_seeds_list, src_len_list
	


	def global_to_local(self):
		
		sub_in_nids = self.src_nids_list
		# print('src global')
		# print(sub_in_nids)#----------------
		# global_nid_2_local = {sub_in_nids[i]: i for i in range(0, len(sub_in_nids))}
		global_nid_2_local = dict(zip(sub_in_nids,range(len(sub_in_nids))))
		self.local_output_nids = list(map(global_nid_2_local.get, self.output_nids.tolist()))
		# print('dst local')
		# print(self.local_output_nids)#----------------
		self.local_src_nids = list(map(global_nid_2_local.get, self.src_nids_list))
		
		self.local=True
		return 	


	def local_to_global(self):
		sub_in_nids = self.src_nids_list
		# local_nid_2_global = { i: sub_in_nids[i] for i in range(0, len(sub_in_nids))}
		local_nid_2_global = dict(zip(range(len(sub_in_nids)), sub_in_nids))
		global_batched_seeds_list=[]
		for local_in_nids in self.local_batched_seeds_list:
			global_in_nids = list(map(local_nid_2_global.get, local_in_nids))
			global_batched_seeds_list.append(global_in_nids)

		self.global_batched_seeds_list=global_batched_seeds_list
		# print('-----------------------------------------------global batched output nodes id----------------------------')
		# for inp in self.global_batched_seeds_list:
		# 	print(len(sorted(inp)))
		# print(self.global_batched_seeds_list)
		self.local=False
		return 


	def init_graph_partition(self):
		t1 = time.time()
		self.global_to_local() # global to local            self.local_batched_seeds_list
		print('global_to_local spend time (sec)', (time.time()-t1))
		
		# Then, the graph_parition is run in block to graph local nids,it has no relationship with raw graph
		self.graph_partition()
		# after that, we transfer the nids of batched output nodes from local to global.
		t2 = time.time()
		self.local_to_global() # local to global         self.global_batched_seeds_list
		print("local_to_global spend time (sec)", (time.time() - t2))

		return self.global_batched_seeds_list, self.weights_list, self.partition_len_list