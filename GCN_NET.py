import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.function as fn
from dgl.nn.pytorch import GraphConv
from GCN_layer import GCNLayer
from mlp_readout_layer import MLPReadout



class GCNNet(nn.Module):
    
    def __init__(self, net_params):
        super().__init__()

        in_dim_node = net_params['in_dim'] # node_dim (feat is an integer)
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        n_output = net_params['n_output']
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']
        n_layers = net_params['L']
        self.readout = net_params['readout']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        self.n_output = n_output
        self.device = net_params['device']
        
        self.embedding_h = nn.Embedding(in_dim_node, hidden_dim) # node feat is an integer
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        self.layers = nn.ModuleList([GCNLayer(hidden_dim, hidden_dim, F.relu, dropout,
                                              self.batch_norm, self.residual) for _ in range(n_layers-1)])
        self.layers.append(GCNLayer(hidden_dim, out_dim, F.relu, dropout, self.batch_norm, self.residual))
        self.MLP_layer = MLPReadout(out_dim, n_output)        


    def forward(self, g, h, e):

        # input embedding
        h = self.embedding_h(h)
        h = self.in_feat_dropout(h)
        
        # GCN
        for conv in self.layers:
            h = conv(g, h)

        # output
        h_out = self.MLP_layer(h)

        return h_out

    
    def loss(self, scores, targets):
        # loss = nn.MSELoss()(scores,targets)
        loss = nn.L1Loss()(scores, targets)
        return 

if __name__ == '__main__':
    
    net_params = {}
    net_params['in_dim'] = 4
    net_params['hidden_dim'] = 128
    net_params['out_dim'] = 128
    net_params['n_output'] = 2
    net_params['in_feat_dropout'] = 0.
    net_params['dropout'] = 0.
    net_params['L'] = 2
    net_params['readout'] = True
    net_params['graph_norm'] = True
    net_params['batch_norm'] = True
    net_params['residual'] = True
    net_params['device'] = 'cuda'

    net = GCNNet(net_params)
    print(net)
