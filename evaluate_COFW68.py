import argparse
import numpy as np
import cv2
import torch
from easydict import EasyDict as edict
from scipy.integrate import simps
from tqdm import tqdm

from libs.models.networks.models import LandmarkModel
from libs.utils.metrics import compute_nme
from libs.utils.image import mean_std_normalize
from model_config import *
from libs.dataset.cofw_dataset import COFWDataset


def parse_args():
    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument("-i", "--images", required=True, help="Path to image folder for training")
    parser.add_argument("--annotation", required=True, help="Annotation file for training")
    parser.add_argument("--in_memory", action="store_true", help="Load all image on RAM")
    parser.add_argument("--image_size", default=256, type=int)
    parser.add_argument("--show", action="store_true")

    # save config
    parser.add_argument("-s", "--weights", default="saved_models/300W_weights/robust-planet-208_epoch_27.pt",
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
    running_nme_hm = 0
    metrics_graph = {"test": [0, 0, 0]}
    errors = []
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
            errors.append(nme_graph)
            if nme_graph > 0.1:
                metrics_graph["test"][1] += 1

        total_nme, num_failures, count = metrics_graph['test']
        print("num_samples: {}\nNME: {}".format(count, total_nme / count))
        AUCError(errors)
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
    dataset = COFWDataset(args.annotation,
                          args.images,
                          image_size=(args.image_size, args.image_size),
                          keypoint_label_names=keypoint_label_names,
                          downsampling_factor=net.hm_model.downsampling_factor,
                          in_memory=args.in_memory,
                          crop_face_storing="temp/test",
                          radius=4,
                          normalize_func=mean_std_normalize,
                          hrnet_box=True)
    run_evaluation(net, dataset, device)


if __name__ == '__main__':
    args = parse_args()
    main(args)
