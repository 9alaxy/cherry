import __init__
import torch
from gcn_lib.sparse.torch_vertex import GENConv
from gcn_lib.sparse.torch_nn import norm_layer
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import logging


class DeeperGCN(torch.nn.Module):
    def __init__(self, args):
        super(DeeperGCN, self).__init__()

        self.num_layers = 14
        self.dropout = args.dropout
        self.block = 'res+'

        self.checkpoint_grad = False

        in_channels = args.in_feats
        hidden_channels = args.num_hidden
        num_tasks = args.n_classes
        conv = 'gen'
        aggr = 'softmax_sg'

        t = 0.1
        self.learn_t = False
        p = 1.0
        self.learn_p = False
        y = 0.0
        self.learn_y = False

        self.msg_norm = False
        learn_msg_scale = True

        norm = 'batch'
        mlp_layers = 1

        if aggr in ['softmax_sg', 'softmax', 'power'] and self.num_layers > 7:
            self.checkpoint_grad = True
            self.ckp_k = self.num_layers // 2

        print('The number of layers {}'.format(self.num_layers),
              'Aggregation method {}'.format(aggr),
              'block: {}'.format(self.block))

        if self.block == 'res+':
            print('LN/BN->ReLU->GraphConv->Res')
        elif self.block == 'res':
            print('GraphConv->LN/BN->ReLU->Res')
        elif self.block == 'dense':
            raise NotImplementedError('To be implemented')
        elif self.block == "plain":
            print('GraphConv->LN/BN->ReLU')
        else:
            raise Exception('Unknown block Type')

        self.gcns = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()

        self.node_features_encoder = torch.nn.Linear(in_channels, hidden_channels)
        self.node_pred_linear = torch.nn.Linear(hidden_channels, num_tasks)

        for layer in range(self.num_layers):

            if conv == 'gen':
                gcn = GENConv(hidden_channels, hidden_channels,
                              aggr=aggr,
                              t=t, learn_t=self.learn_t,
                              p=p, learn_p=self.learn_p,
                              y=y, learn_y=self.learn_y,
                              msg_norm=self.msg_norm, learn_msg_scale=learn_msg_scale,
                              norm=norm, mlp_layers=mlp_layers)
            else:
                raise Exception('Unknown Conv Type')

            self.gcns.append(gcn)
            self.norms.append(norm_layer(norm, hidden_channels))

    def forward(self, blocks, x):
        for index, block in enumerate(blocks):
            # 从 DGLGraph 中提取 edge_index
            edge_index = block.edges()
            # 

            # 如果 edge_index 是 tuple (u, v)，将其转换为 torch.Tensor
            if isinstance(edge_index, tuple):
                edge_index = torch.stack(edge_index)

            edge_index = edge_index.long()

            # # 将 feat 作为 x 输入
            # x = feat

            # 以下是原始的计算逻辑，保持不变
            if index == 0:
                h = self.node_features_encoder(x)

            if self.block == 'res+':
                h = self.gcns[0](h, edge_index)

                if self.checkpoint_grad:
                    for layer in range(1, self.num_layers):
                        h1 = self.norms[layer - 1](h)
                        h2 = F.relu(h1)
                        h2 = F.dropout(h2, p=self.dropout, training=self.training)

                        if layer % self.ckp_k != 0:
                            res = checkpoint(self.gcns[layer], h2, edge_index)
                            h = res + h
                        else:
                            h = self.gcns[layer](h2, edge_index) + h

                else:
                    for layer in range(1, self.num_layers):
                        h1 = self.norms[layer - 1](h)
                        h2 = F.relu(h1)
                        h2 = F.dropout(h2, p=self.dropout, training=self.training)
                        h = self.gcns[layer](h2, edge_index) + h

                h = F.relu(self.norms[self.num_layers - 1](h))
                h = F.dropout(h, p=self.dropout, training=self.training)

            elif self.block == 'res':
                h = F.relu(self.norms[0](self.gcns[0](h, edge_index)))
                h = F.dropout(h, p=self.dropout, training=self.training)

                for layer in range(1, self.num_layers):
                    h1 = self.gcns[layer](h, edge_index)
                    h2 = self.norms[layer](h1)
                    h = F.relu(h2) + h
                    h = F.dropout(h, p=self.dropout, training=self.training)

            elif self.block == 'dense':
                raise NotImplementedError('To be implemented')

            elif self.block == 'plain':
                h = F.relu(self.norms[0](self.gcns[0](h, edge_index)))
                h = F.dropout(h, p=self.dropout, training=self.training)

                for layer in range(1, self.num_layers):
                    h1 = self.gcns[layer](h, edge_index)
                    h2 = self.norms[layer](h1)
                    h = F.relu(h2)
                    h = F.dropout(h, p=self.dropout, training=self.training)
            else:
                raise Exception('Unknown block Type')

            h = self.node_pred_linear(h)

        return torch.log_softmax(h, dim=-1)
    
    def reset_parameters(self):
        return

    def print_params(self, epoch=None, final=False):

        if self.learn_t:
            ts = []
            for gcn in self.gcns:
                ts.append(gcn.t.item())
            if final:
                print('Final t {}'.format(ts))
            else:
                logging.info('Epoch {}, t {}'.format(epoch, ts))

        if self.learn_p:
            ps = []
            for gcn in self.gcns:
                ps.append(gcn.p.item())
            if final:
                print('Final p {}'.format(ps))
            else:
                logging.info('Epoch {}, p {}'.format(epoch, ps))

        if self.learn_y:
            ys = []
            for gcn in self.gcns:
                ys.append(gcn.sigmoid_y.item())
            if final:
                print('Final sigmoid(y) {}'.format(ys))
            else:
                logging.info('Epoch {}, sigmoid(y) {}'.format(epoch, ys))

        if self.msg_norm:
            ss = []
            for gcn in self.gcns:
                ss.append(gcn.msg_norm.msg_scale.item())
            if final:
                print('Final s {}'.format(ss))
            else:
                logging.info('Epoch {}, s {}'.format(epoch, ss))

