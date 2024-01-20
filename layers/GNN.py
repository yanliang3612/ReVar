import torch
from torch_geometric.nn import BatchNorm, GCNConv, LayerNorm, SAGEConv, Sequential, APPNP,GATConv,SAGEConv,SGConv,ChebConv
import torch.nn as nn
import torch.nn.functional as F
from src.argument import parse_args

class GNN(nn.Module):
    def __init__(self, layer_sizes, batchnorm_mm=0.99, args=None):
        super().__init__()

        self.input_size, self.representation_size = layer_sizes[0], layer_sizes[-1]
        layers = []
        if args.net == 'SGC':
            in_dim, out_dim = self.input_size, self.representation_size
            layers.append((SGConv(in_dim, out_dim, K=int(len(layer_sizes) - 1), cached=True), 'x, edge_index -> x'), )
            layers.append(BatchNorm(out_dim, momentum=batchnorm_mm))
            layers.append(nn.PReLU())
        else:
            for in_dim, out_dim in zip(layer_sizes[:-1], layer_sizes[1:]):
                # layers.append( (nn.Dropout(drop_rate), 'x -> x'), )
                if args.net == 'GCN' :
                    layers.append((GCNConv(in_dim, out_dim), 'x, edge_index -> x'), )
                elif args.net == 'GAT' :
                    layers.append((GATConv(in_dim, out_dim // args.n_head, heads=args.n_head), 'x, edge_index -> x'), )
                elif args.net == 'SAGE':
                    layers.append((SAGEConv(in_dim, out_dim), 'x, edge_index -> x'), )
                elif args.net == 'CHEB':
                    layers.append((ChebConv(in_dim, out_dim,args.chebgcn_para), 'x, edge_index -> x'), )

                layers.append(BatchNorm(out_dim, momentum=batchnorm_mm))

                layers.append(nn.PReLU())

        self.model = Sequential('x, edge_index', layers)


    def forward(self, data):
        return self.model(data.x, data.edge_index)

    def reset_parameters(self):
        self.model.reset_parameters()