import numpy as np
import torch
from easydict import EasyDict as edict
# from libs.models.losses import heatmap_loss
# from libs.utils.metrics import compute_nme
from libs.models.networks.hourglass import Hourglass, StackedHourglass
from libs.models.networks.models import HGLandmarkModel
from libs.models.networks.graph_connectivity_model import GCNLandmark


# create network
dims = [[256, 256, 384], [384, 384, 512]]
graph_model_configs = None          # TODO combine these configs
config = {"num_classes": 98,
          "embedding_hidden_sizes": [32],
          "class_embedding_size": 1,
          "edge_hidden_size": 4,
          "visual_feature_dim": 1920,
          "visual_hidden_sizes": [512, 128, 32],
          "visual_embedding_size": 8,
          "GCN_dims": [8, 4],
          "self_connection": False,
          # "graph_norm": "softmax",
          "graph_norm": "mean",
          "device": torch.device("cuda"),
          }
config = edict(config)
device = torch.device("cuda")


net = HGLandmarkModel(3, config.num_classes, dims, graph_model_configs, device,
                      include_graph_model=False,
                      downsample=4)
gcn = GCNLandmark(config).to(device)

input_ = torch.rand([2, 3, 512, 512]).to(device)
batch_size = len(input_)
classIds = torch.arange(config.num_classes).to(device)
xs = torch.randint(128, (config.num_classes,)).to(device)
ys = torch.randint(128, (config.num_classes,)).to(device)
node_locations = torch.stack([xs, ys], dim=1)
with torch.no_grad():
    hm_batch = net(input_)
    visual_features_batch = []
    for loc in node_locations:
        visual_features_batch.append(net.stackedHG.pooling_feature(loc).reshape(batch_size, 1, config.visual_feature_dim))
    visual_features_batch = torch.cat(visual_features_batch, dim=1)

    for i in range(batch_size):
        hm = hm_batch[i]
        node_confidence = hm[classIds, ys, xs]
        out = gcn(node_locations, node_confidence, visual_features_batch[i])
print("finished")

