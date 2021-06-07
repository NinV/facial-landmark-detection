"""
Train CNN backbone only
"""
import argparse
import numpy as np
import cv2
import torch
from easydict import EasyDict as edict
from scipy.integrate import simps
from tqdm import tqdm
import pandas as pd

from libs.dataset.wflw_dataset import WFLWDataset
from libs.models.losses import heatmap_loss
from libs.models.networks.models import LandmarkModel
from libs.utils.metrics import compute_nme
from libs.utils.image import mean_std_normalize, reverse_mean_std_normalize
from model_config import *


def parse_args():
    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument("-i", "--images", required=True, help="Path to image folder for training")
    parser.add_argument("--annotation", required=True, help="Annotation file for training")
    parser.add_argument("--in_memory", action="store_true", help="Load all image on RAM")
    parser.add_argument("--image_size", default=256, type=int)
    parser.add_argument("--show", action="store_true")

    # save config
    parser.add_argument("-s", "--weights", default="saved_models/graph_base_line/frosty-spaceship-175-epoch_19.pt",
                        help="path to saved weights")
    return parser.parse_args()


def AUCError(errors, failureThreshold=0.1, step=0.0001):
    nErrors = len(errors)
    xAxis = list(np.arange(0., failureThreshold + step, step))
    ced = [float(np.count_nonzero([errors <= x])) / nErrors for x in xAxis]
    AUC = simps(ced, x=xAxis) / failureThreshold
    failureRate = 1. - ced[-1]
    print("AUC @ {0}: {1}".format(failureThreshold, AUC))
    print("Failure rate: {0}".format(failureRate))


def run_evaluation(net, dataset, device):
    net.eval()
    # running_nme_hm = 0
    metrics_graph = {"test": [0, 0, 0], "pose": [0, 0, 0], "expression": [0, 0, 0],
                         "illumination": [0, 0, 0], "make-up": [0, 0, 0], "occlusion": [0, 0, 0], "blur": [0, 0, 0]}
    subset_names = ["pose", "expression", "illumination", "make-up", "occlusion", "blur"]
    errors = {"test": [], "pose": [], "expression": [],
              "illumination": [], "make-up": [], "occlusion": [], "blur": []}

    csv_headers = []
    for i in range(98):
        csv_headers.extend(("x{}".format(i), "y{}".format(i)))
    csv_headers.extend(("x_min_rect", "y_min_rect", "x_max_rect", "y_max_rect"))
    csv_headers.extend(("pose", "expression", "illumination", "make-up", "occlusion", "blur"))
    csv_headers.append("image_name")
    df = pd.read_csv(args.annotation, names=csv_headers, sep=" ")
    subset_type = df.loc[:, ["pose", "expression", "illumination", "make-up", "occlusion", "blur"]]

    with torch.no_grad():
        net.eval()
        for i, data in tqdm(enumerate(dataset), total=len(dataset)):
            img, gt_kps, _, _ = data
            img_tensor = torch.unsqueeze(img, 0).to(device, dtype=torch.float)
            gt_kps = np.expand_dims(gt_kps, axis=0) * net.hm_model.downsampling_factor

            # pred_hm_tensor = net(img_tensor)
            pred_hm_tensor, pred_kps_graph = net(img_tensor)
            pred_kps_graph = pred_kps_graph.cpu()
            batch_size, num_classes, h, w = pred_hm_tensor.size()
            hm_size = torch.tensor([h, w])
            pred_kps_graph *= (hm_size * net.hm_model.downsampling_factor)

            meta = {'pts': torch.tensor(gt_kps[:, :, :2])}
            nme_graph = np.sum(compute_nme(pred_kps_graph, meta), keepdims=False)

            metrics_graph["test"][0] += nme_graph
            metrics_graph["test"][2] += 1
            errors["test"].append(nme_graph)
            if nme_graph > 0.1:
                metrics_graph["test"][1] += 1

            category = subset_type.iloc[i].to_list()
            for j, value in enumerate(category):
                if value == 1:
                    metrics_graph[subset_names[j]][0] += nme_graph
                    metrics_graph[subset_names[j]][2] += 1
                    errors[subset_names[j]].append(nme_graph)
                    if nme_graph > 0.1:
                        metrics_graph[subset_names[j]][1] += 1

        for subset, (total_nme, num_failures, count) in metrics_graph.items():
            print("Subset {}, num_samples: {}\nNME: {}".format(subset, count, total_nme / count))
            AUCError(errors[subset])
            print("-----------------")

        return metrics_graph


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create network
    net = LandmarkModel(heatmap_model_config, edict(graph_model_config), device, use_hrnet=True)
    if args.weights:
        print("Load pretrained weight at:", args.weights)
        net.load_state_dict(torch.load(args.weights))

    net.eval()
    keypoint_label_names = list(range(heatmap_model_config["num_classes"]))
    dataset = WFLWDataset(args.annotation,
                          args.images,
                          image_size=(args.image_size, args.image_size),
                          keypoint_label_names=keypoint_label_names,
                          downsampling_factor=net.hm_model.downsampling_factor,
                          in_memory=args.in_memory,
                          crop_face_storing="temp/train",
                          radius=2,
                          normalize_func=mean_std_normalize,
                          hrnet_box=True
                          )
    run_evaluation(net, dataset, device)


if __name__ == '__main__':
    args = parse_args()
    main(args)
