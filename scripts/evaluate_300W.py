import argparse
import pathlib
import sys

project_path = str(pathlib.Path(__file__).absolute().parents[1])
sys.path.insert(0, project_path)

import numpy as np
import torch
from easydict import EasyDict as edict
from scipy.integrate import simps
from tqdm import tqdm

from libs.dataset.w300_dataset import W300_Dataset
from libs.models.networks.models import LandmarkModel
from libs.utils.metrics import compute_nme
from libs.utils.image import mean_std_normalize

heatmap_model_config = {"in_channels": 3,
                        "num_classes": 68,
                        "hg_dims": [[256, 256, 384], [384, 384, 512]],
                        "downsample": True
                        }

graph_model_config = {"num_classes": 68,
                      "embedding_hidden_sizes": [32],
                      "class_embedding_size": 1,
                      "edge_hidden_size": 4,
                      # "visual_feature_dim": 1920,     # Stacked Hourglass
                      "visual_feature_dim": 270,  # HRNet18
                      "visual_hidden_sizes": [512, 128, 32],
                      "visual_embedding_size": 8,
                      "GCN_dims": [64, 16],
                      "self_connection": False,
                      "graph_norm": "softmax",
                      }


def parse_args():
    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument("--annotation", required=True, help="Annotation file for training")
    parser.add_argument("--in_memory", action="store_true", help="Load all image on RAM")
    parser.add_argument("--image_size", default=256, type=int)
    parser.add_argument("--show", action="store_true")

    # save config
    parser.add_argument("-w", "--weights", default="saved_models/300W_weights/robust-planet-208_epoch_27.pt",
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
    return failureRate, AUC


def run_evaluation(net, dataset, device, subset_name="test"):
    print("Evaluating subset {}".format(subset_name))
    net.eval()
    # running_nme_hm = 0
    metrics_graph = {subset_name: [0, 0, 0]}
    errors = {subset_name: []}

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
            pred_kps_hm = net.hm_model.decode_heatmap(pred_hm_tensor, confidence_threshold=-0.01)
            pred_kps_hm *= net.hm_model.downsampling_factor

            meta = {'pts': torch.tensor(gt_kps[:, :, :2])}
            nme_graph = np.sum(compute_nme(pred_kps_graph, meta), keepdims=False)

            metrics_graph[subset_name][0] += nme_graph
            metrics_graph[subset_name][2] += 1
            errors[subset_name].append(nme_graph)
            if nme_graph > 0.1:
                metrics_graph[subset_name][1] += 1

        for subset, (total_nme, num_failures, count) in metrics_graph.items():
            print("Subset {}, num_samples: {}\nNME: {}".format(subset, count, total_nme / count))
            fr, auc = AUCError(errors[subset])
            nme = total_nme / count
            print("-----------------")

        return nme, auc, fr


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create network
    net = LandmarkModel(heatmap_model_config, edict(graph_model_config), device, use_hrnet=True)
    if args.weights:
        print("Load pretrained weight at:", args.weights)
        net.load_state_dict(torch.load(args.weights))

    net.eval()
    keypoint_label_names = list(range(heatmap_model_config["num_classes"]))
    root_folder = pathlib.Path(args.annotation)

    test_common = W300_Dataset(str(root_folder / "test_common"),
                               str(root_folder / "test_common"),
                               image_size=(args.image_size, args.image_size),
                               keypoint_label_names=keypoint_label_names,
                               downsampling_factor=net.hm_model.downsampling_factor,
                               in_memory=args.in_memory,
                               crop_face_storing="temp/test_common",
                               normalize_func=mean_std_normalize,
                               hrnet_box=True)

    test_challenge = W300_Dataset(str(root_folder / "test_challenge"),
                                  str(root_folder / "test_challenge"),
                                  image_size=(args.image_size, args.image_size),
                                  keypoint_label_names=keypoint_label_names,
                                  downsampling_factor=net.hm_model.downsampling_factor,
                                  in_memory=args.in_memory,
                                  crop_face_storing="temp/test_challenge",
                                  normalize_func=mean_std_normalize,
                                  hrnet_box=True)

    test_official = W300_Dataset(str(root_folder / "300W_test"),
                                 str(root_folder / "300W_test"),
                                 image_size=(args.image_size, args.image_size),
                                 keypoint_label_names=keypoint_label_names,
                                 downsampling_factor=net.hm_model.downsampling_factor,
                                 in_memory=args.in_memory,
                                 crop_face_storing="temp/test_challenge",
                                 normalize_func=mean_std_normalize,
                                 hrnet_box=True)

    nme1, auc1, fr1 = run_evaluation(net, test_common, device, subset_name="common")
    nme2, auc2, fr2 = run_evaluation(net, test_challenge, device, subset_name="challenge")

    common, challenge = len(test_common), len(test_challenge)
    nme = (nme1 * common + nme2 * challenge) / (common + challenge)
    auc = (auc1 * common + auc2 * challenge) / (common + challenge)
    fr = (fr1 * common + fr2 * challenge) / (common + challenge)
    print("Subset {}, num_samples: {}\nNME: {}\nAUC0.1: {}\nFR0.1: {}".format("full",
                                                                              common + challenge, nme, auc, fr))

    run_evaluation(net, test_official, device, subset_name="official test")


if __name__ == '__main__':
    args = parse_args()
    main(args)
