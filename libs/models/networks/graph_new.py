import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GraphModel(nn.Module):
    def __init__(self,input_dim=100,output_dim=2,hidden_node_dim=[256,112,64,32]):
        super(GraphModel, self).__init__()
        self.hidden = [input_dim,*hidden_node_dim,output_dim]
        self.W = [ nn.Linear( self.hidden[i] , self.hidden[i + 1] , bias=True ) for i in range(len(self.hidden)-1)]
        print(self.W)
        self.E = nn.Linear(2 , 1 , bias=True )
        self.relu = nn.ReLU(inplace=True)

    def forward(self,node_feature, edge_feature):
        '''
            node_feature dim (nxm) feature dim (nxm) : n number node, m number feature
            edge_feature dim (nx1) 

        '''

        Edge = torch.zeros((edge_feature.shape[0],edge_feature.shape[0]))
        for i in range(edge_feature.shape[0]):
            for j in range(edge_feature.shape[0]):
                if i!=j :
                    feature =  torch.tensor((edge_feature[i],edge_feature[j]))
                    Edge[i,j] = self.E(feature)
        h = node_feature
        for i in range(len(self.W)):
            h = self.relu(self.W[i](h))
            for i in range(h.shape[0]):
                node_i = h[i,:]
                for j in range(Edge.shape[0]):
                    node_i += Edge[i,j].item()*h[i,:] 
            h[i,:]= node_i
        return h

if __name__ == '__main__':
    net = GraphModel()
    node_feature =  torch.rand((256,100))
    edge_feature = torch.rand((256,1))
    x = net(node_feature,edge_feature)
    print(x.shape)
        
