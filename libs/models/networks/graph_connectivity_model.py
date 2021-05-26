# TODO: change this code to work with batch of more than 1 with batch matmaul

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassEmbedding(nn.Module):
    def __init__(self, num_classes, hidden_sizes, embedding_size):
        super(ClassEmbedding, self).__init__()
        self.num_classes = num_classes
        layers = []
        current_dim = num_classes * 2
        for h in hidden_sizes:
            layers.append(nn.Linear(current_dim, h))
            current_dim = h
        self.hidden_layers = nn.ModuleList(layers)
        self.out = nn.Linear(current_dim, embedding_size)

    def forward(self, x):
        """
        x : (num_pairs, 2)
        """
        x = F.one_hot(x, num_classes=self.num_classes).view(-1, 2 * self.num_classes).float()
        for layer in self.hidden_layers:
            x = layer(x)
            x = F.relu(x)
        x = self.out(x)
        return torch.sigmoid(x)


class EdgeWeights(nn.Module):
    def __init__(self, embedding_size=1, hidden=4):
        super(EdgeWeights, self).__init__()
        self.linear = nn.Linear(embedding_size + 2, hidden)
        self.out = nn.Linear(hidden, 1)

    def forward(self, x):
        x = self.linear(x)
        x = F.relu(x, inplace=True)
        x = self.out(x)
        # return torch.sigmoid(x)
        return x


class VisualFeatureEmbedding(nn.Module):
    def __init__(self, in_channels, hidden_dims, embedding_size):
        super(VisualFeatureEmbedding, self).__init__()
        self.in_channels = in_channels
        layers = []
        current_dim = in_channels
        for h in hidden_dims:
            layers.append(nn.Linear(current_dim, h))
            current_dim = h
        self.hidden_layers = nn.ModuleList(layers)
        self.out = nn.Linear(current_dim, embedding_size)

    def forward(self, x):
        for layer in self.hidden_layers:
            x = layer(x)
            x = F.relu(x)
        x = self.out(x)
        return x


class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features, self_connection=False):
        super(GCNLayer, self).__init__()
        self.in_features, self.out_features = in_features, out_features
        self.self_connection = self_connection

        self.w1 = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.kaiming_uniform_(self.w1, a=math.sqrt(5))

        if not self_connection:
            self.w2 = torch.nn.Parameter(torch.Tensor(out_features, in_features))
            nn.init.kaiming_uniform_(self.w2, a=math.sqrt(5))

    def forward(self, x, edges):
        """
        :param x: (num_classes, in_features)
        :param edges: (num_classes, in_features), should be transpose before putting into this function because
        we're using F.linear instead of matmul

        using F.linear instead of matmul may improve performance:
        https://discuss.pytorch.org/t/why-does-the-linear-module-seems-to-do-unnecessary-transposing/6277/2
        https://stackoverflow.com/questions/18796801/increasing-the-data-locality-in-matrix-multiplication

        Consider change to torch.BMM in case: x.size() = (batch_size, num_classes, in_features)
        and edges.size() = (batch_size, num_classes, in_features)
        """
        device = self.w1.device
        num_nodes = x.size()[0]
        out = torch.empty((num_nodes, self.out_features), device=device)
        for i in range(num_nodes):
            messages = F.linear(x, self.w1)
            messages = edges[i].view(-1, 1) * messages
            aggregate = torch.sum(messages, dim=0)
            if self.self_connection:
                out[i] = aggregate
            else:
                target_node = x[i]
                out[i] = F.linear(target_node, self.w1) + aggregate
        return out


class GCNLandmark(nn.Module):

    def __init__(self, config, device=torch.device("cuda")):
        super(GCNLandmark, self).__init__()
        self.num_classes = config.num_classes
        self.device = device
        self.self_connection = config.self_connection
        if config.graph_norm == "softmax":
            self.graph_norm = torch.softmax
        elif config.graph_norm == "mean":
            self.graph_norm = torch.mean
        else:
            self.graph_norm = None

        self.class_embedding = ClassEmbedding(config.num_classes, config.embedding_hidden_sizes,
                                              config.class_embedding_size)
        self.edges = EdgeWeights(config.class_embedding_size, config.edge_hidden_size)
        self.visual_feature_embedding = VisualFeatureEmbedding(config.visual_feature_dim, config.visual_hidden_sizes,
                                                               config.visual_embedding_size)

        self.gcn_dims = config.GCN_dims
        gcn_layers = []
        current_dim = config.visual_embedding_size + 2  # +2 for 2D location
        for h in self.gcn_dims:
            gcn_layers.append(GCNLayer(current_dim, h, self.self_connection))
            current_dim = h
        gcn_layers.append(GCNLayer(current_dim, 2, self.self_connection))
        self.gcn_layers = torch.nn.ModuleList(gcn_layers)

        self.to(self.device)
        self.pairs = self._generate_node_pairs()

    def _generate_node_pairs(self):
        target_nodes_indices = list(range(self.num_classes))
        pairs = []
        for i in target_nodes_indices:
            if self.self_connection:
                neighbor_nodes_indices = target_nodes_indices
            else:
                neighbor_nodes_indices = target_nodes_indices[:i] + target_nodes_indices[i+1:]
            for j in neighbor_nodes_indices:
                pairs.append([i, j])
        return torch.tensor(pairs).to(self.device)

    def forward(self, node_positions, node_confidences, visual_features):
        """
        :param node_positions: [num_nodes, 2] - should be normalized
        :param node_confidences: [num_nodes, 1]
        :param visual_features: [num_nodes, dims]
        """
        # constructing edges matrix
        class_embedding = self.class_embedding(self.pairs)
        X = node_confidences[self.pairs.reshape(-1)].reshape(-1, 2)
        X = torch.cat([X, class_embedding], dim=1)

        neighbor_edges = self.edges(X).view(-1)

        if self.self_connection:
            edges_full = neighbor_edges.view(self.num_classes, self.num_classes)
        else:
            edges_full = torch.zeros((self.num_classes, self.num_classes), device=self.device, dtype=torch.float)
            target_nodes_indices, neighbor_nodes_indices = self.pairs[:, 0], self.pairs[:, 1]
            edges_full[target_nodes_indices, neighbor_nodes_indices] = neighbor_edges

        # normalizing edges
        edges_full = self.graph_norm(edges_full, dim=1)
        # edges_full.transpose_(1, 0)     # transpose this edges because we're using F.linear instead of matmul

        # construct node features
        visual_embedding = self.visual_feature_embedding(visual_features)
        node_features = torch.cat([node_positions, visual_embedding], dim=1)

        # GCN forward
        x = node_features
        for layer in self.gcn_layers:
            x = layer(x, edges_full)
        return x
