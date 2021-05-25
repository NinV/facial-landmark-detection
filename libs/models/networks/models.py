import torch
import torch.nn as nn
from libs.models.networks.hourglass import HGLandmarkModel
from libs.models.networks.graph_connectivity_model import GCNLandmark


class LandmarkModel(nn.Module):
    def __init__(self, hm_model_config, gcn_config, mode, device="cuda"):
        """
        :param mode:
                    "fine_tune_graph": freeze heatmap model and train GCN model
                    "train_both": train both heatmap and GCN models
                    "inference": inference mode
        """
        super(LandmarkModel, self).__init__()
        self.mode = mode
        self.device = device
        self.hm_model = HGLandmarkModel(**hm_model_config, device=device).to(self.device)
        self.gcn_model = GCNLandmark(gcn_config).to(self.device)

    def forward(self, x):
        hm = self.hm_model(x)
        kps_from_hm = self.hm_model.decode_heatmap(hm, confidence_threshold=0)  # (batch_size, num_classes, 3)
        batch_size, num_classes, h, w = hm.size()
        hm_size = torch.tensor([h, w])
        node_positions = kps_from_hm[:, :, :2]      # (batch_size, num_classes, 2)
        out = []
        for i in range(batch_size):
            visual_features = []
            for loc in node_positions[i]:
                visual_features.append(self.hm_model.stackedHG.pooling_feature(i, loc))
            visual_features = torch.stack(visual_features, dim=0)

            xs = node_positions[i, :, 0]
            ys = node_positions[i, :, 1]
            classIds = torch.arange(num_classes)
            node_confidences = hm[i, classIds, ys, xs]
            node_positions_normalized = (node_positions[i] / hm_size).to(self.device)
            out.append(self.gcn_model(node_positions_normalized, node_confidences, visual_features))

        return torch.stack(out, dim=0)
