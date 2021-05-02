
from libs.models.networks.hourglass import *


def make_pre_layer(in_channels, pre_dims=(128, 256)):
    layer = nn.Sequential(convolution(7, in_channels, pre_dims[0], stride=2),
                          residual(pre_dims[0], pre_dims[1], stride=2))
    downsampling_factor = 4
    out_dim = pre_dims[-1]
    return layer, downsampling_factor, out_dim


class HGLandmarkModel(nn.Module):
    def __init__(self, in_channels, hg_dims, num_classes):
        super(HGLandmarkModel, self).__init__()
        self.num_classes = num_classes

        self.downsample, self.downsampling_factor, current_dim = make_pre_layer(in_channels)
        self.stackedHG = StackedHourglass(current_dim, hg_dims)

        # heatmap prediction
        self.hm = nn.Sequential(nn.Conv2d(hg_dims[1][0], num_classes, kernel_size=3, padding=1, stride=1),
                                nn.Sigmoid())

    def forward(self, x):
        x = self.downsample(x)
        x = self.stackedHG(x)
        hm = self.hm(x)

        with torch.no_grad():
            kps_from_heatmap = self.decode_heatmap(hm, 0.)

        node_features = []
        for i in range(len(kps_from_heatmap)):
            node_features.append([])
            for x, y, classId in kps_from_heatmap[i]:
                f = torch.cat([torch.tensor([x, y]), hm[i, :, y, x]])
                node_features[i].append(f)

        node_features = torch.tensor(node_features)

        # construct node features
        return hm

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
                    indices_y, indices_x = torch.nonzero(hm_reduce == torch.max(hm_reduce[i, c]))
                    if hm_reduce[i, c, indices_y, indices_x] > confidence_threshold:
                        kps[i].append([indices_x, indices_y, c])
        return kps
