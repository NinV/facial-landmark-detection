import dgl
from libs.models.networks.hourglass import *
from libs.models.networks.GCN_NET import get_graph_model


def make_pre_layer(in_channels, pre_dims=(128, 256)):
    layer = nn.Sequential(convolution(7, in_channels, pre_dims[0], stride=2),
                          residual(pre_dims[0], pre_dims[1], stride=2))
    downsampling_factor = 4
    out_dim = pre_dims[-1]
    return layer, downsampling_factor, out_dim


class HGLandmarkModel(nn.Module):
    def __init__(self, in_channels, num_classes, hg_dims, graph_model_configs, device="cuda",
                 downsample=False, include_graph_model=True):
        super(HGLandmarkModel, self).__init__()
        self.num_classes = num_classes
        self.device = device
        self.graph_model_configs = graph_model_configs
        self.include_graph_model = include_graph_model

        if downsample:
            self.downsample, self.downsampling_factor, current_dim = make_pre_layer(in_channels)
            self.stackedHG = StackedHourglass(current_dim, hg_dims)
        else:
            self.downsample = None
            self.downsampling_factor = 1
            self.stackedHG = StackedHourglass(in_channels, hg_dims)

        # heatmap prediction
        self.hm = nn.Sequential(nn.Conv2d(hg_dims[1][0], num_classes, kernel_size=3, padding=1, stride=1),
                                nn.Sigmoid())
        if self.include_graph_model:
            self.graph_model = get_graph_model()
        self.to(self.device)

    def forward(self, x):
        if self.downsample is not None:
            x = self.downsample(x)
        x = self.stackedHG(x)
        hm = self.hm(x)
        if not self.include_graph_model:
            return hm
        with torch.no_grad():
            kps_from_heatmap = self.decode_heatmap(hm, 0.)
        node_features = []
        for i in range(len(kps_from_heatmap)):
            node_features.append([])
            for x, y, classId in kps_from_heatmap[i]:
                f = torch.cat([torch.tensor([x, y], device=self.device), hm[i, :, y, x]])
                node_features[i].append(f)
            node_features[i] = torch.stack(node_features[i])

        node_features = torch.stack(node_features)
        graphs = self._connecting_node(node_features[:, ])

        graph_model_outputs = []
        for g, h in zip(graphs, node_features):
            g = g.to(self.device)
            graph_model_outputs.append(self.graph_model(g, h, None))

        graph_model_outputs = torch.stack(graph_model_outputs)
        return hm, graphs, graph_model_outputs

    @staticmethod
    def decode_heatmap(hm, confidence_threshold=0.2, kernel=3, one_landmark_per_class=True):
        """
        hm : pytorch tensor of shape (num_samples, c, h, w)
        """
        pad = (kernel - 1) // 2
        hmax = torch.nn.functional.max_pool2d(hm, (kernel, kernel), stride=1, padding=pad)
        keep = (hmax == hm).float()
        hm_reduce = (hm * keep)
        # find indices
        kps = []
        batch_size, num_classes, h, w = hm.size()
        for i in range(batch_size):
            kps.append([])
            for c in range(num_classes):
                if one_landmark_per_class:
                    # https://discuss.pytorch.org/t/get-indices-of-the-max-of-a-2d-tensor/82150
                    indices_y, indices_x = torch.nonzero(hm_reduce[i, c] == torch.max(hm_reduce[i, c]), as_tuple=True)
                    indices_x, indices_y = indices_x[0], indices_y[0]  # ensure only there only one landmark per class
                    if hm_reduce[i, c, indices_y, indices_x] > confidence_threshold:
                        kps[i].append([indices_x, indices_y, c])
        return torch.tensor(kps)

    def _connecting_node(self, node_pos):
        configs = self.graph_model_configs
        if configs["nodes_connecting"] == "topk":
            graphs = []
            for i in range(len(node_pos)):
                graphs.append(dgl.knn_graph(node_pos[i], configs["k"]))
            return graphs

        if configs["nodes_connecting"] == "delaunay":
            raise NotImplementedError

        if configs["nodes_connecting"] == "fully_connected":
            raise NotImplementedError

        else:
            raise ValueError("nodes_connecting mode is not recognized")


