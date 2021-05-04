import sys
from pathlib import Path
project_path = str(Path(__file__).absolute().parents[1])
sys.path.insert(0, project_path)

import torch
from libs.models.networks.models import HGLandmarkModel


dims = [[256, 256, 384], [384, 384, 512]]
graph_model_configs = {"nodes_connecting": "topk",
                       "k": 2}
"""
        if configs["nodes_connecting"] == "topk":
            graphs = []
            for i in range(len(node_pos)):
                graphs.append(dgl.knn_graph(node_pos[i], configs["topk"]))
            return graphs
"""
image_size = 512, 512

device = "cuda"     # or device="cpu"
# model = StackedHourglass(3, dims).to(device=device)
model = HGLandmarkModel(3, 15, dims, graph_model_configs, device=device)

# reduce memory usage
with torch.no_grad():
    image = torch.rand(2, 3, *image_size).to(device=device)
    hm, graphs, graph_model_outputs = model(image)

print("finished")
