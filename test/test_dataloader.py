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
from libs.dataset.coco_dataset import KeypointDataset
from libs.dataset.wflw_dataset import WFLWDataset
from libs.utils.augmentation import SequentialTransform, RandomScalingAndRotation, RandomTranslation, ColorDistortion

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
    parser.add_argument("--num_classes", default=98, type=int, help="Number of landmark classes")
    parser.add_argument("--image_size", default=512, type=int)

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


def get_augmentation(args):
    translation = RandomTranslation((-0.1, 0.1), (-0.1, 0.1))
    rotation_and_scaling = RandomScalingAndRotation((-10, 10), (0.8, 1.2))
    color_distortion = ColorDistortion()
    # blurring = GaussianBlur(0.5)
    transform = SequentialTransform([translation, rotation_and_scaling], [0.5, 0.5],
                                    [color_distortion], [0.5],
                                    (args.image_size, args.image_size))
    return transform


def main(args):
    transform = get_augmentation(args)
    keypoint_label_names = list(range(args.num_classes))
    dataset = WFLWDataset(args.annotation,
                          args.images,
                          image_size=(args.image_size, args.image_size),
                          downsampling_factor=1,
                          in_memory=False,
                          radius=args.radius,
                          keypoint_label_names=keypoint_label_names,
                          normalize_func=None,
                          augmentation=transform,
                          force_square_shape=False)
    print("finished loading")
    print("Num images:", len(dataset), dataset.images, dataset._image_ids)
    # for j in range(5):
    for i, (img, kps, hm, transform_params) in enumerate(dataset):
        img = img.permute(1, 2, 0).numpy()
        hm = hm.permute(1, 2, 0).numpy()
        hm = np.max(hm, axis=-1)
        img = img.astype(np.uint8)
        for (x, y, classId) in kps:
            cv2.circle(img, (int(x + 0.5), int(y + 0.5)), radius=2, thickness=-1, color=[0, 255, 0])

        cv2.imshow("img", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.imshow("hm", (hm * 255).astype(np.uint8))
        k = cv2.waitKey(0)
        if k == ord("q"):
            break


if __name__ == '__main__':
    args = parse_args()
    main(args)
