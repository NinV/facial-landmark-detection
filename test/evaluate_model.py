import argparse
import pathlib
from time import time
import sys
from pathlib import Path
project_path = str(Path(__file__).absolute().parents[1])
sys.path.insert(0, project_path)

import cv2
import numpy as np
import torch
from torch.utils.data import random_split

from libs.models.networks.hourglass import StackedHourglass
from libs.dataset.dataset import KeypointDataset
from libs.models.losses import heatmap_loss


def parse_args():
    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument("-i", "--images", required=True, help="Path to image folder")
    parser.add_argument("--annotation", required=True, help="Annotation file (.json)")
    parser.add_argument("--in_memory", action="store_true", help="Load all image on RAM")
    parser.add_argument("--split", type=float, default=0.9, help="Train-Test split ratio")
    parser.add_argument("--seed", type=int, default=42, help="random seed for train-test split")
    parser.add_argument("--test_set", type=int, choices=[1, 2], required=True, help="1: training set, 2: test set")
    parser.add_argument("--radius", type=int, default=4)

    # save config
    parser.add_argument("-s", "--saved_model", default="saved_models/HG_best_train.pt", help="folder for saving model")
    parser.add_argument("--save_best_only", action="store_true", help="only save best weight")
    parser.add_argument("--valid_interval", type=int, default=2, help="Save model and evaluate interval")
    return parser.parse_args()


def create_folder(path):
    path = pathlib.Path(path)
    path.mkdir(parents=True, exist_ok=True)


def run_single_prediction(net, inputs, labels):
    with torch.no_grad():
        outputs = net(inputs)
        loss = heatmap_loss(outputs, labels)
    print("Testing loss:", loss.item(), end="\n-----------------------------------------------------------\n\n")
    return outputs


def decode_heatmap(hm, kernel=3, num_classes=15):
    """
    hm : (h, w, c) numpy array
    """
    pad = (kernel - 1) // 2
    hmax = torch.nn.functional.max_pool2d(hm, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == hm).float()
    hm_reduce = (hm * keep)[0]
    print(hm.shape)
    print(hm_reduce.shape)
    hm_reduce = hm_reduce.permute(1, 2, 0)
    return hm_reduce.detach().cpu().numpy()


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create network
    dims = [[256, 256, 384], [384, 384, 512]]
    net = StackedHourglass(3, dims, 15).to(device)
    net.load_state_dict(torch.load(args.saved_model))
    net.eval()

    keypoint_label_names = list(range(15))
    dataset = KeypointDataset(args.annotation,
                              args.images,
                              keypoint_label_names=keypoint_label_names,
                              downsampling_factor=4,
                              in_memory=False,
                              radius=args.radius)

    num_training = int(len(dataset) * args.split)
    num_testing = len(dataset) - num_training
    training_set, test_set = random_split(dataset, [num_training, num_testing],
                                          generator=torch.Generator().manual_seed(args.seed))

    train_loader = torch.utils.data.DataLoader(training_set, batch_size=1, shuffle=False, drop_last=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, drop_last=False)

    # with torch.no_grad():
    if args.test_set == 1:
        loader = train_loader
    else:
        loader = test_loader

    for i, data in enumerate(loader):
        inputs, labels = data
        inputs = inputs.to(device, dtype=torch.float)
        labels = labels.to(device, dtype=torch.float)
        print("inputs shape:", inputs.shape)
        pred = run_single_prediction(net, inputs, labels)
        print("prediction shape:", pred.shape)
        # gt = decode_heatmap(labels[0]) * 255
        pred = decode_heatmap(pred)
        pred = np.max(pred, axis=2, keepdims=False) * 255

        # cv2.imshow("gt", gt.astype(np.uint8))
        cv2.imshow("pred", pred.astype(np.uint8))
        k = cv2.waitKey(0)
        if k == ord("q"):
            break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    args = parse_args()
    main(args)
