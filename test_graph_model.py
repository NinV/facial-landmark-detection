import torch
from easydict import EasyDict as edict
# from libs.models.losses import heatmap_loss
# from libs.utils.metrics import compute_nme
from libs.models.networks.hourglass import Hourglass, StackedHourglass
from libs.models.networks.models import HGLandmarkModel
from libs.models.networks.graph_connectivity_model import GCNLandmark


config = {"num_classes": 98,
          "embedding_hidden_sizes": [32],
          "class_embedding_size": 1,
          "edge_hidden_size": 4,
          "device": torch.device("cuda"),
          "self_connection": True
          }
config = edict(config)
device = torch.device("cuda")

# create network
dims = [[256, 256, 384], [384, 384, 512]]
graph_model_configs = None
net = HGLandmarkModel(3, config.num_classes, dims, graph_model_configs, device,
                      include_graph_model=False,
                      downsample=4)
gcn = GCNLandmark(config).to(device)

input_ = torch.rand([2, 3, 512, 512]).to(device)

# loc1 = 0, 50, 100
# loc2 = 1, 20, 30
classIds = torch.arange(config.num_classes).to(device)
xs = torch.randint(128, (config.num_classes,)).to(device)
ys = torch.randint(128, (config.num_classes,)).to(device)
# landmarks = torch.stack([classIds, ys, xs], dim=1)
with torch.no_grad():
    hm_batch = net(input_)
    for i in range(len(input_)):
        hm = hm_batch[i]
        node_confidence = hm[classIds, ys, xs]
        class_embedding, X, edges = gcn(node_confidence, None)

        print(edges[:3, 0:3])

print("finished")
