"""
Train CNN backbone only
"""
import argparse
import pathlib

import torch
from torch.utils.data import random_split
import wandb

from libs.models.networks.models import HGLandmarkModel
from libs.dataset.coco_dataset import KeypointDataset
from libs.dataset.wflw_dataset import WFLWDataset
from libs.models.losses import heatmap_loss
from libs.utils.heatmap import decode_heatmap
from libs.utils.metrics import normalized_mean_error


def parse_args():
    parser = argparse.ArgumentParser()

    # model config
    parser.add_argument("--num_classes", default=98, type=int, help="Number of landmark classes")
    parser.add_argument("--image_size", default=512, type=int)
    parser.add_argument("--downsample", action="store_false", help="Disable downsampling")

    # dataset
    parser.add_argument("-i", "--images", required=True, help="Path to image folder for training")
    parser.add_argument("--annotation", required=True, help="Annotation file for training")
    parser.add_argument("--test_images", default="",
                        help="Path to image folder for testing. If not given, split from training set")
    parser.add_argument("--test_annotation", default="",
                        help="Annotation file for testing. If not given, split from training set")
    parser.add_argument("--split", type=float, default=0.9, help="Train-Test split ratio")
    parser.add_argument("--seed", type=int, default=42, help="random seed for train-test split")
    parser.add_argument("--format", default="WFLW", help="dataset format: 'WFLW', 'COCO'")
    parser.add_argument("--in_memory", action="store_true", help="Load all image on RAM")
    parser.add_argument("-n", "--normalized_index", nargs='+', type=int, default=[96, 97],
                        help="landmarks indexes for calculate normalize distance in NME")

    # save config
    parser.add_argument("-s", "--saved_folder", default="saved_models", help="folder for saving model")
    parser.add_argument("--save_best_only", action="store_true", help="only save best weight")
    parser.add_argument("--valid_interval", type=int, default=2, help="Save model and evaluate interval")

    # training hyperparameters
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--radius", type=int, default=4)
    parser.add_argument("--lr", type=float, default=10e-4)
    parser.add_argument("--mode", help="Training mode: 0 - heatmap only, 1 - graph model")
    # parser.add_argument("--learning_rate", type=float, default=10e-3, help="Initial learning rate")
    # parser.add_argument("--decay_steps", type=float, default=10000, help="learning rate decay step")
    # parser.add_argument("--decay_rate", type=float, default=0.995, help="learning rate decay rate")
    # parser.add_argument("--staircase", action="store_true", help="learning rate decay on step (default: smooth)")
    return parser.parse_args()


def create_folder(path):
    path = pathlib.Path(path)
    path.mkdir(parents=True, exist_ok=True)


def train_one_epoch(net, optimizer, loader, epoch, device, args):
    net.train()
    for i, data in enumerate(loader):
        img, gt_kps, gt_hm, _ = data
        num_samples = img.size()[0]
        img = img.to(device, dtype=torch.float)
        gt_hm = gt_hm.to(device, dtype=torch.float)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        pred_hm = net(img)
        loss = heatmap_loss(pred_hm, gt_hm)
        loss.backward()
        optimizer.step()
        print("batch {}/{}, heat map loss: {}".format(i + 1, len(loader),
                                                      loss.item()))
        wandb.log({'train_total_loss (step)': loss.item(),
                   'epoch': epoch,
                   'batch': i + 1})


def run_evaluation(net, loader, epoch, device, prefix='val'):
    net.eval()
    running_hm_loss = 0
    with torch.no_grad():
        for i, data in enumerate(loader):
            img, gt_kps, gt_hm, _ = data
            img = img.to(device, dtype=torch.float)
            gt_hm = gt_hm.to(device, dtype=torch.float)
            gt_kps = gt_kps.to(device, dtype=torch.float)
            pred_hm = net(img)
            hm_loss = heatmap_loss(pred_hm, gt_hm)
            running_hm_loss += hm_loss.item() * len(img)

    gt_kps = gt_kps.detach().cpu().tolist()
    pred_kps = net.decode_heatmap(pred_hm, confidence_threshold=0.0)
    pred_kps = pred_kps.detach().cpu().tolist()
    nme, _, _ = normalized_mean_error(gt_kps, pred_kps, args.normalized_index)

    num_samples = len(loader.dataset)
    running_hm_loss /= num_samples
    wandb.log({'{}_hm_loss'.format(prefix): running_hm_loss,
               '{}_nme'.format(prefix): nme,
               'epoch': epoch})

    return running_hm_loss, nme


def main(args):
    create_folder(args.saved_folder)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create network
    dims = [[256, 256, 384], [384, 384, 512]]
    # graph_model_configs = {"nodes_connecting": "topk",
    #                        "k": 2}
    graph_model_configs = None
    net = HGLandmarkModel(3, args.num_classes, dims, graph_model_configs, device,
                          include_graph_model=False, downsample= args.downsample)

    keypoint_label_names = list(range(args.num_classes))
    if args.format == "COCO":
        dataset = KeypointDataset(args.annotation,
                                  args.images,
                                  image_size = (args.image_size, args.image_size),
                                  keypoint_label_names=keypoint_label_names,
                                  downsampling_factor=net.downsampling_factor,
                                  in_memory=args.in_memory,
                                  radius=args.radius)
    elif args.format == "WFLW":
        dataset = WFLWDataset(args.annotation,
                              args.images,
                              image_size=(args.image_size, args.image_size),
                              keypoint_label_names=keypoint_label_names,
                              downsampling_factor=net.downsampling_factor,
                              in_memory=args.in_memory,
                              crop_face_storing="temp/train",
                              radius=args.radius,
                              )
    else:
        raise ValueError("Wrong data format")

    if args.test_annotation:
        training_set = dataset
        dataset_type = type(training_set)
        test_set = dataset_type(args.test_annotation,
                                args.test_images,
                                image_size=(args.image_size, args.image_size),
                                keypoint_label_names=keypoint_label_names,
                                downsampling_factor=net.downsampling_factor,
                                in_memory=args.in_memory,
                                crop_face_storing="temp/test",
                                radius=args.radius)
    else:
        num_training = int(len(dataset) * args.split)
        num_testing = len(dataset) - num_training
        training_set, test_set = random_split(dataset, [num_training, num_testing],
                                              generator=torch.Generator().manual_seed(args.seed))

    train_loader = torch.utils.data.DataLoader(training_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
    eval_train_loader = torch.utils.data.DataLoader(training_set, batch_size=args.batch_size * 2, drop_last=False)
    eval_test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size * 2, drop_last=False)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    training_config = {"batch_size": args.batch_size,
                       "epochs": args.epochs,
                       "radius": args.radius,
                       "learning_rate": args.lr,
                       "image_size": args.image_size,
                       "num_classes": args.num_classes,
                       "normalized_index": args.normalized_index,
                       "downsample": args.downsample}
    if args.test_images:
        training_config["split"] = args.split
        training_config["seed"] = args.seed
    wandb.init(project="gnn-landmarks",
               config=training_config)

    curr_train_nme = float("inf")
    curr_valid_nme = float("inf")
    for epoch in range(1, args.epochs + 1):  # loop over the dataset multiple times
        print("Training epoch", epoch)
        train_one_epoch(net, optimizer, train_loader, epoch, device, args)

        print("Evaluating on training set")
        train_hm_loss, train_nme = run_evaluation(net, eval_train_loader, epoch, device)
        print("hm loss: {}, NME: {}".format(train_hm_loss, train_nme),
              end="\n-------------------------------------------\n\n")
        print("Evaluating on testing set")
        val_hm_loss, val_nme = run_evaluation(net, eval_test_loader, epoch, device)
        print("hm loss: {}, NME: {}".format(val_hm_loss, val_nme),
              end="\n-------------------------------------------\n\n")

        if train_nme < curr_train_nme:
            torch.save(net.state_dict(), "{}/HG_best_train.pt".format(args.saved_folder))
            curr_train_nme = train_nme

        if val_nme < curr_valid_nme:
            torch.save(net.state_dict(), "{}/HG_best_val.pt".format(args.saved_folder))
            curr_valid_nme = val_nme

    torch.save(net, "{}/HG_final.pt".format(args.saved_folder))
    print('Finished Training')
    return net


if __name__ == '__main__':
    args = parse_args()
    main(args)
