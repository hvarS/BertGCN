"""GCN using DGL nn package
References:
- Semi-Supervised Classification with Graph Convolutional Networks
- Paper: https://arxiv.org/abs/1609.02907
- Code: https://github.com/tkipf/gcn
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
# from dgl.nn import GraphConv
from .graphconv_edge_weight import GraphConvEdgeWeight as GraphConv
import torch_geometric.nn as gnn

class GCN(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 normalization='none'):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GraphConv(in_feats, n_hidden, activation=activation, norm=normalization))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GraphConv(n_hidden, n_hidden, activation=activation, norm=normalization))
        # output layer
        self.layers.append(GraphConv(n_hidden, n_classes, norm=normalization))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, features, g, edge_weight):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h, edge_weights=edge_weight)
        return h

class GCNTorch(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        super(GCNTorch, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.layers = nn.ModuleList()
        self.layers.append(gnn.GCNConv(in_feats, n_hidden))
        self.layers.append(activation())
        self.dropout = nn.Dropout(p=dropout)
        # hidden layers
        for _ in range(n_layers - 1):
            self.layers.append(gnn.GCNConv(n_hidden, n_hidden))
            self.layers.append(activation())
            self.layers.append(self.dropout)
        # output layer
        self.layers.append(gnn.GCNConv(n_hidden, n_classes))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, features, edge_index, edge_weight):
        h = features
        for i, layer in enumerate(self.layers):
            print(h.device,edge_index.device,edge_weight.device)
            if i%3==0:
                h = layer(h,edge_index =edge_index,edge_weight=edge_weight)
            else:
                h = layer(h) 
        return h