import torch
from torch import nn
import cv2
import numpy as np

from libs.models.networks.hourglass import StackedHourglass
from libs.models.losses import heatmap_loss
from libs.dataset.dataset import KeypointDataset
from libs.models.losses import heatmap_loss


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dims = [[256, 256, 384], [384, 384, 512]]
    net = StackedHourglass(3, dims, 15).to(device)
    net.eval()

    single_tensor = np.random.rand(1, 3, 512, 512)
    double_tensor = np.repeat(single_tensor, 2, axis=0)
    sample_gt = np.random.rand(1, 15, 128, 128)

    single_tensor = torch.from_numpy(single_tensor).float().to(device)
    double_tensor = torch.from_numpy(double_tensor).float().to(device)
    sample_gt = torch.from_numpy(sample_gt).float().to(device)

    with torch.no_grad():
        out1 = net(single_tensor)
        loss1 = heatmap_loss(out1, sample_gt)

        out2 = net(double_tensor)
        loss2 = heatmap_loss(out2, sample_gt)

        print(loss1.item(), loss2.item())


if __name__ == '__main__':
    main()
