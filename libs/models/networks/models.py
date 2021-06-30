import torch
import torch.nn as nn
from libs.models.networks.hourglass import HGLandmarkModel
from libs.models.networks.graph_connectivity_model import GCNLandmark
from libs.models.networks.HRNet import get_face_alignment_net as get_HR_model


class LandmarkModel(nn.Module):
    def __init__(self, hm_model_config, gcn_config, device="cuda", use_hrnet=False,
                 freeze_hm_model=False, hrnet_config='face_alignment_300w_hrnet_w18.yaml'):
        """
        :param mode:
                    "fine_tune_graph": freeze heatmap model and train GCN model
                    "train_both": train both heatmap and GCN models
                    "inference": inference mode
        """
        super(LandmarkModel, self).__init__()
        self.freeze_hm_model = freeze_hm_model
        self.device = device
        if use_hrnet:
            self.hm_model = get_HR_model(hrnet_config).to(self.device)
        else:
            self.hm_model = HGLandmarkModel(**hm_model_config, device=device).to(self.device)
        self.gcn_model = GCNLandmark(gcn_config).to(self.device)

    def forward(self, x):
        if self.freeze_hm_model:
            self.hm_model.eval()
            with torch.no_grad():
                hm = self.hm_model(x)
        else:
            hm = self.hm_model(x)

        kps_from_hm = self.hm_model.decode_heatmap(hm, confidence_threshold=0)  # (batch_size, num_classes, 3)
        batch_size, num_classes, h, w = hm.size()
        hm_size = torch.tensor([h, w])
        node_positions = kps_from_hm[:, :, :2]      # (batch_size, num_classes, 2)
        out = []
        for i in range(batch_size):
            visual_features = []
            for loc in node_positions[i]:
                visual_features.append(self.hm_model.pooling_feature(i, loc))
            visual_features = torch.stack(visual_features, dim=0)

            xs = node_positions[i, :, 0]
            ys = node_positions[i, :, 1]
            classIds = torch.arange(num_classes)
            node_confidences = hm[i, classIds, ys, xs]
            node_positions_normalized = (node_positions[i] / hm_size).to(self.device)
            out.append(self.gcn_model(node_positions_normalized, node_confidences, visual_features))

        return hm, torch.stack(out, dim=0)
