import argparse
import pathlib
import numpy as np
import torch
from PIL import Image
from easydict import EasyDict as edict
from scipy.integrate import simps
from tqdm import tqdm

from libs.dataset.w300_dataset import W300_Dataset
from libs.models.networks.models import LandmarkModel
from libs.utils.metrics import compute_nme
from libs.utils.image import mean_std_normalize, reverse_mean_std_normalize
from model_config import *


class ResizeNotPadding:
    def __init__(self, image_size, training=True):
        self.image_size = image_size
        self.training = training

    def __call__(self, img, kps):
        # resized_img, ratio, (dw, dh) = letterbox(img, new_shape=(self.h, self.w), auto=not self.training)
        # kps_resized = np.asarray(kps, dtype=np.float)
        # kps_resized[:, :2] *= ratio  # ratio = [ratio_w, ratio_h]
        # kps_resized[:, 0] += dw
        # kps_resized[:, 1] += dh
        h, w = img.shape[:2]
        nw, nh = self.image_size
        resized_img = np.array(Image.fromarray(img).resize(self.image_size))
        scale_x = nw / w
        scale_y = nh / h
        kps_resized = np.asarray(kps, dtype=np.float)
        kps_resized[:, :2] *= (scale_x,scale_y)

        return resized_img, kps_resized, [scale_x, scale_y, 0, 0]

    @staticmethod
    def inverse_resize(kps, ratio, dw, dh):
        if not isinstance(kps, np.ndarray):
            kps_ = np.asarray(kps, dtype=np.float)
        else:
            kps_ = kps.copy()
        kps_[:, 0] -= dw
        kps_[:, 1] -= dh
        kps_[:, :2] /= ratio
        return kps_

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


def run_evaluation(net, dataset, device):
    net.eval()
    # running_nme_hm = 0
    metrics_graph = {"test": [0, 0, 0]}
    errors = {"test": []}

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

            metrics_graph["test"][0] += nme_graph
            metrics_graph["test"][2] += 1
            errors["test"].append(nme_graph)
            if nme_graph > 0.1:
                metrics_graph["test"][1] += 1

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
    # test_common.resize_func = ResizeNotPadding((256, 256))
    # test_challenge.resize_func = ResizeNotPadding((256, 256))
    nme1, auc1, fr1 = run_evaluation(net, test_common, device)
    nme2, auc2, fr2 = run_evaluation(net, test_challenge, device)

    common, challenge = len(test_common), len(test_challenge)
    nme = (nme1 * common + nme2 * challenge) / (common + challenge)
    auc = (auc1 * common + auc2 * challenge) / (common + challenge)
    fr = (fr1 * common + fr2 * challenge) / (common + challenge)
    print("Subset {}, num_samples: {}\nNME: {}\nAUC0.1: {}\nFR0.1: {}".format("full",
                                                                              common + challenge, nme, auc, fr))


if __name__ == '__main__':
    args = parse_args()
    main(args)
