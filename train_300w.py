"""
Train CNN backbone only
"""
import argparse
import pathlib

from easydict import EasyDict as edict
import numpy as np
import torch
from torch.utils.data import random_split
import wandb

from libs.models.networks.models import LandmarkModel
from libs.dataset.wflw_dataset import WFLWDataset
from libs.dataset.cofw_dataset import COFWDataset
from libs.dataset.w300_dataset import W300_Dataset
from libs.models.losses import heatmap_loss
from libs.utils.metrics import compute_nme
from libs.utils.augmentation import SequentialTransform, RandomScalingAndRotation, RandomTranslation, ColorDistortion, \
    HorizontalFlip
from libs.utils.image import mean_std_normalize
from model_config import heatmap_model_config, graph_model_config


def parse_args():
    parser = argparse.ArgumentParser()

    # weights config
    parser.add_argument("--weights", default="", help="load weights")
    parser.add_argument("--model", default="full", help="specific loaded model type in: ['backbone', 'full'] ")

    # dataset
    parser.add_argument("--annotation", required=True, help="Annotation file for training")
    parser.add_argument("--seed", type=int, default=42, help="random seed for train-test split")
    parser.add_argument("--in_memory", action="store_true", help="Load all image on RAM")
    parser.add_argument("--image_size", default=256, type=int)

    # save config
    parser.add_argument("-s", "--saved_folder", default="saved_models", help="folder for saving model")
    parser.add_argument("--save_best_only", action="store_true", help="only save best weight")
    parser.add_argument("--valid_interval", type=int, default=2, help="Save model and evaluate interval")

    # training hyperparameters
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--radius", type=int, default=4)
    parser.add_argument("--lr", type=float, default=10e-4, help="backbone learning rate")
    parser.add_argument("--gcn_lr", type=float, default=10e-3, help="gcn learning rate")
    parser.add_argument("--mode", help="Training mode: 0 - heatmap only, 1 - graph only, 2 - both")
    parser.add_argument("--regression_loss", default="L1", help="'L1' or 'L2'")
    parser.add_argument("--multi_gpu", action="store_true")
    parser.add_argument("--freeze_hm", action="store_true")

    # augmentation
    parser.add_argument("--augmentation", action="store_true")
    parser.add_argument("--tx", nargs='+', type=float, default=[-0.1, 0.1])
    parser.add_argument("--ty", nargs='+', type=float, default=[-0.1, 0.1])
    parser.add_argument("--t_prob", type=float, default=0.5)
    parser.add_argument("--rot", nargs='+', type=float, default=[-10, 10])
    parser.add_argument("--scale", nargs='+', type=float, default=[0.8, 1.2])
    parser.add_argument("--rot_and_scale_prob", type=float, default=0.5)
    parser.add_argument("--color", type=float, default=0.5)
    parser.add_argument("--hue", type=float, default=0.2)
    parser.add_argument("--saturation", type=float, default=1.5)
    parser.add_argument("--exposure", type=float, default=1.5)

    return parser.parse_args()


def create_folder(path):
    path = pathlib.Path(path)
    path.mkdir(parents=True, exist_ok=True)


def get_augmentation(args):
    translation = RandomTranslation(args.tx, args.ty)
    rotation_and_scaling = RandomScalingAndRotation(args.rot, args.scale)
    hflip = HorizontalFlip()
    color_distortion = ColorDistortion(hue=args.hue, saturation=args.saturation, exposure=args.exposure)
    transform = SequentialTransform([translation, rotation_and_scaling, hflip],
                                    [args.t_prob, args.rot_and_scale_prob, 0.5],
                                    [color_distortion], [args.color],
                                    (args.image_size, args.image_size))
    return transform


def train_one_epoch(net, optimizer, loader, epoch, device, opt):
    for i, data in enumerate(loader):
        img, gt_kps, gt_hm, _ = data
        img = img.to(device, dtype=torch.float)
        gt_hm = gt_hm.to(device, dtype=torch.float)

        batch_size, num_classes, h, w = gt_hm.size()
        hm_size = torch.tensor([h, w])
        gt_kps[:, :, :2] /= hm_size
        gt_kps = gt_kps.to(device, dtype=torch.float)

        # zero the parameter gradients
        optimizer.zero_grad()

        # heatmap loss
        pred_hm, pred_kps = net(img)
        hm_loss = heatmap_loss(pred_hm, gt_hm)
        # hm_loss = torch.nn.MSELoss(reduction="mean")(pred_hm, gt_hm)

        #  regression loss
        if opt.regression_loss == 'L1':
            regression_loss = torch.nn.L1Loss(reduction="mean")(pred_kps, gt_kps[:, :, :2])
        else:
            regression_loss = torch.nn.MSELoss(reduction="mean")(pred_kps, gt_kps[:, :, :2])

        loss = hm_loss + regression_loss
        loss.backward()
        optimizer.step()
        print("batch {}/{}, heat map loss: {}, regression loss: {}".format(i + 1, len(loader),
                                                                           hm_loss.item(), regression_loss.item()))
        wandb.log({'train_hm_loss (step)': hm_loss.item(),
                   'train_regr_loss (step)': regression_loss.item(),
                   'epoch': epoch,
                   'batch': i + 1})


def run_evaluation(net, loader, epoch, device, opt, prefix='val'):
    net.eval()
    running_hm_loss = 0
    running_regression_loss = 0
    running_nme_hm = 0
    running_nme_graph = 0
    with torch.no_grad():
        for i, data in enumerate(loader):
            img, gt_kps, gt_hm, _ = data
            img = img.to(device, dtype=torch.float)
            gt_hm = gt_hm.to(device, dtype=torch.float)

            gt_kps = gt_kps.to(device, dtype=torch.float)
            pred_hm, pred_kps_graph = net(img)

            hm_loss = heatmap_loss(pred_hm, gt_hm)
            running_hm_loss += (hm_loss.item() * len(img))

            #  regression loss
            batch_size, _, _, _ = pred_hm.size()
            if opt.regression_loss == 'L1':
                regression_loss = torch.nn.L1Loss(reduction="mean")(pred_kps_graph, gt_kps[:, :, :2])
            else:
                regression_loss = torch.nn.MSELoss(reduction="mean")(pred_kps_graph, gt_kps[:, :, :2])
            running_regression_loss += (regression_loss.item() * batch_size)

            pred_kps_graph = pred_kps_graph.cpu()
            batch_size, num_classes, h, w = pred_hm.size()
            hm_size = torch.tensor([h, w])
            pred_kps_graph *= hm_size

            pred_kps_hm = net.hm_model.decode_heatmap(pred_hm, confidence_threshold=-0.01)
            nme_hm = np.sum(compute_nme(pred_kps_hm[:, :, :2], {'pts': gt_kps[:, :, :2]}), keepdims=False)
            nme_graph = np.sum(compute_nme(pred_kps_graph, {'pts': gt_kps[:, :, :2]}), keepdims=False)
            running_nme_hm += nme_hm
            running_nme_graph += nme_graph
    num_samples = len(loader.dataset)
    running_hm_loss /= num_samples
    running_regression_loss /= num_samples
    running_nme_hm /= num_samples
    running_nme_graph /= num_samples
    wandb.log({'{}_hm_loss'.format(prefix): running_hm_loss,
               '{}_regr_loss'.format(prefix): running_regression_loss,
               '{}_nme (hm)'.format(prefix): running_nme_hm,
               '{}_nme (graph)'.format(prefix): running_nme_graph,
               'epoch': epoch})

    return running_hm_loss, running_nme_graph


def main(args):
    create_folder(args.saved_folder)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = LandmarkModel(heatmap_model_config, edict(graph_model_config), device, use_hrnet=True,
                        freeze_hm_model=args.freeze_hm, hrnet_config="face_alignment_300w_hrnet_w18.yaml")

    net.to(device)

    if args.weights:
        if args.model == "backbone":
            print("Load pretrained backbone weights at:", args.weights)
            net.hm_model.load_state_dict(torch.load(args.weights))
        elif args.model == "full":
            print("Load pretrained full model weights at:", args.weights)
            net.load_state_dict(torch.load(args.weights))
        else:
            raise ValueError("wrong model type")
    else:
        net.hm_model.load_state_dict(torch.load("saved_models/hrnetv2_pretrained/HR18-300W_processed.pth"))

    # data loader
    keypoint_label_names = list(range(68))

    if args.augmentation:
        transform = get_augmentation(args)
    else:
        transform = None

    root_folder = pathlib.Path(args.annotation)
    training_set = W300_Dataset(str(root_folder / "train"),
                                str(root_folder / "train"),
                                image_size=(args.image_size, args.image_size),
                                keypoint_label_names=keypoint_label_names,
                                downsampling_factor=net.hm_model.downsampling_factor,
                                in_memory=None,
                                crop_face_storing='temp/train',
                                radius=args.radius,
                                augmentation=transform,
                                normalize_func=mean_std_normalize,
                                hrnet_box=True)

    test_common = W300_Dataset(str(root_folder / "test_common"),
                               str(root_folder / "test_common"),
                               image_size=(args.image_size, args.image_size),
                               keypoint_label_names=keypoint_label_names,
                               downsampling_factor=net.hm_model.downsampling_factor,
                               in_memory=args.in_memory,
                               crop_face_storing="temp/test_common",
                               radius=args.radius,
                               normalize_func=mean_std_normalize,
                               hrnet_box=True)

    test_challenge = W300_Dataset(str(root_folder / "test_challenge"),
                                  str(root_folder / "test_challenge"),
                                  image_size=(args.image_size, args.image_size),
                                  keypoint_label_names=keypoint_label_names,
                                  downsampling_factor=net.hm_model.downsampling_factor,
                                  in_memory=args.in_memory,
                                  crop_face_storing="temp/test_challenge",
                                  radius=args.radius,
                                  normalize_func=mean_std_normalize,
                                  hrnet_box=True)

    train_loader = torch.utils.data.DataLoader(training_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
    common_set_loader = torch.utils.data.DataLoader(test_common, batch_size=args.batch_size * 2, drop_last=False)
    challenge_set_loader = torch.utils.data.DataLoader(test_challenge, batch_size=args.batch_size * 2, drop_last=False)
    optimizer = torch.optim.Adam([
        {'params': net.hm_model.parameters(), 'lr': args.lr},
        {'params': net.gcn_model.parameters(), 'lr': args.gcn_lr}
    ])

    for epoch in range(1, args.epochs + 1):  # loop over the dataset multiple times
        print("Training epoch", epoch)
        train_one_epoch(net, optimizer, train_loader, epoch, device, args)

        print("Evaluating on testing set")
        val_hm_loss, val_nme = run_evaluation(net, common_set_loader, epoch, device, args, prefix="common")
        print("common set: hm loss: {}, NME: {}".format(val_hm_loss, val_nme),
              end="\n-------------------------------------------\n\n")
        val_hm_loss, val_nme = run_evaluation(net, challenge_set_loader, epoch, device, args, prefix="challenge")
        print("challenge set: hm loss: {}, NME: {}".format(val_hm_loss, val_nme),
              end="\n-------------------------------------------\n\n")

        torch.save(net.state_dict(), "{}/epoch_{}.pt".format(args.saved_folder, epoch))

    print('Finished Training')
    return net


if __name__ == '__main__':
    args = parse_args()
    wandb.init(project="gnn-landmarks",
               config={**vars(args), **heatmap_model_config, **graph_model_config})
    main(args)
