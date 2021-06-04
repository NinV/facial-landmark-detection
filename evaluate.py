"""
Train CNN backbone only
"""
import argparse
import numpy as np
import cv2
import torch
from easydict import EasyDict as edict
from tqdm import tqdm

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
    parser.add_argument("--format", default="WFLW", help="dataset format: 'WFLW', 'COCO'")
    parser.add_argument("--in_memory", action="store_true", help="Load all image on RAM")
    parser.add_argument("--radius", type=int, default=8)
    parser.add_argument("--image_size", default=256, type=int)

    # save config
    parser.add_argument("-s", "--weights", default="saved_models/graph_base_line/frosty-spaceship-175-epoch_19.pt",
                        help="path to saved weights")
    return parser.parse_args()


def plot_kps(img, gt, pred_hm, pred_graph):
    for (x, y, classId) in gt:
        cv2.circle(img, (int(x + 0.5), int(y + 0.5)), radius=2, thickness=-1, color=[0, 255, 0])

    for (x, y, _) in pred_hm:
        cv2.circle(img, (int(x + 0.5), int(y + 0.5)), radius=2, thickness=-1, color=[0, 0, 255])

    for (x, y) in pred_graph:
        cv2.circle(img, (int(x + 0.5), int(y + 0.5)), radius=2, thickness=-1, color=[255, 0, 0])

    return img


def visualize_hm(hm):
    hm = hm.permute(1, 2, 0).cpu().numpy()
    hm = np.max(hm, axis=-1)
    return (hm * 255).astype(np.uint8)


def run_evaluation(net, dataset, device):
    net.eval()
    running_hm_loss = 0
    running_nme = 0
    with torch.no_grad():
        net.eval()
        for i, data in tqdm(enumerate(dataset), total=len(dataset)):
            img, gt_kps, gt_hm, _ = data
            img_tensor = torch.unsqueeze(img, 0).to(device, dtype=torch.float)
            # gt_hm_tensor = torch.unsqueeze(gt_hm, 0).to(device, dtype=torch.float)
            gt_kps = np.expand_dims(gt_kps, axis=0) * net.hm_model.downsampling_factor

            # pred_hm_tensor = net(img_tensor)
            pred_hm_tensor, pred_kps_graph = net(img_tensor)
            pred_kps_graph = pred_kps_graph.cpu()
            batch_size, num_classes, h, w = pred_hm_tensor.size()
            hm_size = torch.tensor([h, w])
            pred_kps_graph *= (hm_size * net.hm_model.downsampling_factor)

            # hm_loss = heatmap_loss(pred_hm_tensor, gt_hm_tensor)
            pred_kps_hm = net.hm_model.decode_heatmap(pred_hm_tensor, confidence_threshold=0.0) * net.hm_model.downsampling_factor

            meta = {'pts': torch.tensor(gt_kps[:, :, :2])}
            nme_hm = np.sum(compute_nme(pred_kps_hm[:, :, :2], meta), keepdims=False)
            nme_graph = np.sum(compute_nme(pred_kps_graph, meta), keepdims=False)
            running_nme += nme_graph
            # print(nme_hm, nme_graph)

            # show image
            pred_kps_hm = pred_kps_hm.detach().cpu().numpy()
            # img = img * 255
            img = img.detach().cpu()
            img = img.permute(1, 2, 0).numpy()
            img = reverse_mean_std_normalize(img).astype(np.uint8)
            img = plot_kps(img, gt_kps[0], pred_kps_hm[0], pred_kps_graph[0])

            gt_hm_np = visualize_hm(gt_hm)
            pred_hm_np = visualize_hm(pred_hm_tensor[0])

            cv2.imshow("img", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            cv2.imshow("gt_hm", gt_hm_np)
            cv2.imshow("pred_hm", pred_hm_np)
            k = cv2.waitKey(0)
            if k == ord("q"):
                break
        print(running_nme / len(dataset))
        return running_nme / len(dataset)


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
                          radius=args.radius,
                          normalize_func=mean_std_normalize,
                          # force_square_shape=True
                          )
    run_evaluation(net, dataset, device)


if __name__ == '__main__':
    args = parse_args()
    main(args)
