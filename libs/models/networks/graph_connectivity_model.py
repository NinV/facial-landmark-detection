import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassEmbedding(nn.Module):
    def __init__(self, num_classes, hidden_sizes=(32, 4), embedding_size=1):
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
        return torch.sigmoid(x)


class GCNLandmark(nn.Module):
    def __init__(self, config):
        super(GCNLandmark, self).__init__()
        self.embedding = ClassEmbedding(config.num_classes, config.embedding_hidden_sizes, config.class_embedding_size)
        self.edges = EdgeWeights(config.class_embedding_size, config.edge_hidden_size)
        self.num_classes = config.num_classes
        self.device = config.device
        self.self_connection = config.self_connection

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

    def forward(self, node_confidences, visual_features):
        """
        :param node_confidences: [num_nodes, 1]
        :param visual_features: [num_nodes, dims]
        """
        # constructing edges matrix
        class_embedding = self.embedding(self.pairs)
        X = node_confidences[self.pairs.reshape(-1)].reshape(-1, 2)
        X = torch.cat([X, class_embedding], dim=1)

        neighbor_edges = self.edges(X).view(-1)

        if self.self_connection:
            edges_full = neighbor_edges.view(self.num_classes, self.num_classes)
        else:
            edges_full = torch.zeros((self.num_classes, self.num_classes), device=self.device, dtype=torch.float)
            target_nodes_indices, neighbor_nodes_indices = self.pairs[:, 0], self.pairs[:, 1]
            edges_full[target_nodes_indices, neighbor_nodes_indices] = neighbor_edges

        return class_embedding, X, edges_full
