import torch
from easydict import EasyDict as edict
from libs.models.networks.models import LandmarkModel


device = torch.device("cuda")
heatmap_mode_config = {"in_channels": 3,
                       "num_classes": 98,
                       "hg_dims": [[256, 256, 384], [384, 384, 512]],
                       "downsample": True,
                       "device": device
                       }

graph_model_configs = {"num_classes": 98,
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
graph_model_configs = edict(graph_model_configs)
net = LandmarkModel(heatmap_mode_config, graph_model_configs, "inference", device)

input_ = torch.rand([2, 3, 512, 512]).to(device)
with torch.no_grad():
    out = net(input_)
print("finished")
