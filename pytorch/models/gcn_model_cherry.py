import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv
import tqdm

class GCN(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, num_layers, activation, dropout):
        self.n_hidden = hidden_feats
        self.n_classes = out_feats
        self.activation = activation
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        if num_layers == 1:
            self.layers.append(GraphConv(in_feats, out_feats))
        else:
            # Input layer
            self.layers.append(GraphConv(in_feats, hidden_feats, activation=activation))
            # Hidden layers
            for _ in range(num_layers - 2):
                self.layers.append(GraphConv(hidden_feats, hidden_feats, activation=activation))
            # Output layer
            self.layers.append(GraphConv(hidden_feats, out_feats))

        self.dropout = nn.Dropout(p=dropout)

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, blocks, features):
        x = features
        if len(self.layers) == 1:
            x = self.layers[0](blocks[0], x)
            x = self.activation(x)
        else:
            for layer, block in zip(self.layers[:-1], blocks):
                with block.local_scope():
                    x = self.dropout(F.relu(layer(block, x)))
            x = self.layers[-1](blocks[-1], x)  # Use the last block for output layer
        return x
    
    def inference(self, g, x, args, device):
        """
        g : the entire graph. dgl graph
        x : the input of entire node set.

        The inference code is written in a fashion that it could handle any number of nodes and
        layers.
        """
        device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
        for l, layer in enumerate(self.layers):
            y = torch.zeros(g.num_nodes(), self.n_hidden if l!=len(self.layers) - 1 else self.n_classes)
            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
            dataloader = dgl.dataloading.DataLoader(
				g,
				torch.arange(g.num_nodes(),dtype=torch.long).to(g.device),
				sampler,
				device=device,
				# batch_size=24,
				batch_size=args.batch_size,
				shuffle=True,
				drop_last=False,
				num_workers=args.num_workers)
            
            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                block = blocks[0]
                block = block.int().to(device)
                input_nodes = input_nodes.to(x.device)
                h = x[input_nodes].to(device)
                h = layer(block, h)
                # y[output_nodes] = h
                y[output_nodes] = h.cpu()
            x = y
        return y


# # Define a simple Graph dataset
# # In a real scenario, you will load your graph data and features accordingly
# g = dgl.graph(([0, 1, 2, 3], [1, 2, 3, 4]))
# features = torch.randn(5, 10)  # Example feature tensor

# # Instantiate GCN model
# in_feats = features.shape[1]
# hidden_feats = 16
# out_feats = 2
# num_layers = 2
# activation = F.relu
# dropout = 0.5
# model = GCN(in_feats, hidden_feats, out_feats, num_layers, activation, dropout)

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
