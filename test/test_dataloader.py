import sys
from pathlib import Path
project_path = str(Path(__file__).absolute().parents[1])
sys.path.insert(0, project_path)

import argparse

import torch
import cv2
import numpy as np
from matplotlib import pyplot as plt

from libs.models.networks.hourglass import StackedHourglass
from libs.dataset.dataset import KeypointDataset


def decode_hm(hm):
    hm = hm.permute(1, 2, 0)
    hm = hm.detach().cpu().numpy()
    hm = np.max(hm, axis=2, keepdims=False)
    return hm


def image_tensor_to_numpy(image_tensor):
    image_np = image_tensor.permute(1, 2, 0)
    image_np = image_np.detach().cpu().numpy()
    return image_np * 255


def parse_args():
    parser = argparse.ArgumentParser()
    # dataset
    parser.add_argument("-i", "--images", required=True, help="Path to image folder")
    parser.add_argument("--annotation", required=True, help="Annotation file (.json)")
    parser.add_argument("--in_memory", action="store_true", help="Load all image on RAM")
    parser.add_argument("--radius", type=int, default=4)

    return parser.parse_args()


def colorize_heatmap(hm_gray):
    colormap = plt.get_cmap('rainbow')
    heatmap_rgb = (colormap(hm_gray) * 2 ** 16).astype(np.uint16)[:, :, :3]
    # heatmap_bgr = cv2.cvtColor(heatmap_rgb, cv2.COLOR_RGB2BGR)
    return heatmap_rgb.astype(np.uint8)


def blend_heat_map(img, hm, alpha=0.5):
    mask = (hm > 0).astype(np.float) * 0.5
    blend = (1 - mask) * img + mask * hm
    return blend


def main(args):
    keypoint_label_names = list(range(15))
    dataset = KeypointDataset(args.annotation,
                              args.images,
                              keypoint_label_names=keypoint_label_names,
                              downsampling_factor=4,
                              in_memory=False,
                              radius=args.radius)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False)
    for i, (inputs, labels) in enumerate(loader):
        hm_gt = decode_hm(labels[0])
        img_rgb = image_tensor_to_numpy(inputs[0])

        h, w = img_rgb.shape[:2]

        hm_gt_resized = cv2.resize(hm_gt, (w, h), interpolation=cv2.INTER_AREA)
        hm_gt_resized_color = colorize_heatmap(hm_gt_resized)
        blend = blend_heat_map(img_rgb, hm_gt_resized_color)

        img = cv2.cvtColor(img_rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
        blend = cv2.cvtColor(blend.astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imshow("gt", hm_gt.astype(np.uint8))
        cv2.imshow("heatmap gray", (hm_gt_resized * 255).astype(np.uint8))
        cv2.imshow("heatmap color", hm_gt_resized_color.astype(np.uint8))
        cv2.imshow("image", img.astype(np.uint8))
        cv2.imshow("blend", blend.astype(np.uint8))
        k = cv2.waitKey(0)
        if k == ord("q"):
            break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    args = parse_args()
    main(args)

