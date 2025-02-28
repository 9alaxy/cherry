import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GATConv
import tqdm

class GAT(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, num_heads, num_layers, activation, dropout):
        self.n_hidden = hidden_feats
        self.n_classes = out_feats
        self.activation = activation
        self.num_heads = num_heads
        super(GAT, self).__init__()
        self.layers = nn.ModuleList()
        if num_layers == 1:
            self.layers.append(GATConv(in_feats, out_feats, num_heads, activation=None, feat_drop=dropout, attn_drop=dropout))
        else:
            # input layer
            self.layers.append(GATConv(in_feats, hidden_feats, num_heads, activation=activation, feat_drop=dropout, attn_drop=dropout))
            # hidden layers
            for _ in range(num_layers - 1):
                self.layers.append(GATConv(hidden_feats * num_heads, hidden_feats, num_heads, activation=activation, feat_drop=dropout, attn_drop=dropout))
            # output layer
            self.layers.append(GATConv(hidden_feats * num_heads, out_feats, num_heads, activation=None, feat_drop=dropout, attn_drop=dropout))

    def forward(self, blocks, features):
        h = features
        for layer, block in zip(self.layers, blocks):
            h = layer(block, h).flatten(1)
            if len(self.layers) == 1:
                h = self.activation(h)
        return h
    
    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()
    
    def inference(self, g, x, args, device):
        device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
        


    # def inference(self, g, x, args, device):
    #     """
    #     g : the entire graph. dgl graph
    #     x : the input of entire node set.

    #     The inference code is written in a fashion that it could handle any number of nodes and
    #     layers.
    #     """
    #     device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    #     for l, layer in enumerate(self.layers):
    #         out_feats_per_head = self.n_classes // self.num_heads
    #         y = torch.zeros(g.num_nodes(), self.n_hidden if l!=len(self.layers) - 1 else self.n_classes)
    #         sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
    #         dataloader = dgl.dataloading.DataLoader(
	# 			g,
	# 			torch.arange(g.num_nodes(),dtype=torch.long).to(g.device),
	# 			sampler,
	# 			device=device,
	# 			# batch_size=24,
	# 			batch_size=args.batch_size,
	# 			shuffle=True,
	# 			drop_last=False,
	# 			num_workers=args.num_workers)
            
    #         for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
    #             block = blocks[0]
    #             block = block.int().to(device)
    #             input_nodes = input_nodes.to(x.device)
    #             h = x[input_nodes].to(device)
    #             h = layer(block, h)
    #             print('input nodes: ', input_nodes.shape)
    #             print('output nodes: ', output_nodes.shape)
    #             print('h: ', h.shape)
    #             y = h.cpu()
    #         x = y
    #     return y



# # Define a simple Graph dataset
# # In a real scenario, you will load your graph data and features accordingly
# g = dgl.graph(([0, 1, 2, 3], [1, 2, 3, 4]))
# features = torch.randn(5, 10)  # Example feature tensor

# # Instantiate GAT model
# in_feats = features.shape[1]
# hidden_feats = 8
# out_feats = 2
# num_heads = [4, 2]  # Number of attention heads in each layer
# num_layers = 2
# activation = F.elu
# dropout = 0.5
# model = GAT(in_feats, hidden_feats, out_feats, num_heads, num_layers, activation, dropout)

# # Define loss function and optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# # Define the training function
# def train(model, g, features, labels, dataloader, optimizer, criterion, epochs):
#     for epoch in range(epochs):
#         total_loss = 0
#         for batched_graph, batch_labels in dataloader:
#             optimizer.zero_grad()
#             output = model(batched_graph, features)
#             loss = criterion(output, batch_labels)
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()
#         print(f'Epoch {epoch+1}, Loss: {total_loss}')

# # Assuming labels are known for each node in the graph
# # In a real scenario, you should have your ground truth labels accordingly
# labels = torch.tensor([0, 1, 0, 1, 0])

# # DataLoader setup
# batch_list = [0, 1, 2, 3, 4]  # Assuming each node is a batch
# sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
# args = {'num_workers': 4}  # Example num_workers setting
# dataloader = dgl.dataloading.DataLoader(
#     g,
#     batch_list,
#     sampler,
#     batch_size=len(batch_list),
#     shuffle=True,
#     drop_last=False,
#     num_workers=args['num_workers']
# )

# # Train the model
# train(model, g, features, labels, dataloader, optimizer, criterion, epochs=10)
