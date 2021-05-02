import argparse
import pathlib

import torch
from torch.utils.data import random_split
import wandb

from libs.models.networks.hourglass import StackedHourglass
from libs.dataset.dataset import KeypointDataset
from libs.models.losses import heatmap_loss
from libs.utils.heatmap import decode_heatmap


def parse_args():
    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument("-i", "--images", required=True, help="Path to image folder")
    parser.add_argument("--annotation", required=True, help="Annotation file (.json)")
    parser.add_argument("--in_memory", action="store_true", help="Load all image on RAM")
    parser.add_argument("--split", type=float, default=0.9, help="Train-Test split ratio")
    parser.add_argument("--seed", type=int, default=42, help="random seed for train-test split")

    # save config
    parser.add_argument("-s", "--saved_folder", default="saved_models", help="folder for saving model")
    parser.add_argument("--save_best_only", action="store_true", help="only save best weight")
    parser.add_argument("--valid_interval", type=int, default=2, help="Save model and evaluate interval")

    # training hyperparameters
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--radius", type=int, default=4)
    parser.add_argument("--lr", type=float, default=10e-4)
    # parser.add_argument("--learning_rate", type=float, default=10e-3, help="Initial learning rate")
    # parser.add_argument("--decay_steps", type=float, default=10000, help="learning rate decay step")
    # parser.add_argument("--decay_rate", type=float, default=0.995, help="learning rate decay rate")
    # parser.add_argument("--staircase", action="store_true", help="learning rate decay on step (default: smooth)")
    return parser.parse_args()


def create_folder(path):
    path = pathlib.Path(path)
    path.mkdir(parents=True, exist_ok=True)


def train_one_epoch(net, optimizer, loader, epoch, device):
    net.train()
    running_loss = 0.0
    for i, data in enumerate(loader):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels, _ = data
        num_samples = inputs.size()[0]
        inputs = inputs.to(device, dtype=torch.float)
        labels = labels.to(device, dtype=torch.float)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = heatmap_loss(outputs, labels) / num_samples
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        print("batch {}/{} loss: {}".format(i+1, len(loader), loss.item()))
        wandb.log({'train_loss': loss.item(), 'epoch': epoch, 'batch': i+1})
    running_loss = running_loss / (i+1)
    print("Training loss:", running_loss)
    return running_loss
    # print("Average loss (across all mini batches):", running_loss/(i+1))
    # torch.save(net.state_dict(), "saved_models/HG_ep_{}.pt".format(epoch+1))


def run_validation(net, loader, epoch, device):
    net.eval()
    with torch.no_grad():
        running_loss = 0.0
        for i, data in enumerate(loader):
            inputs, labels, transform_params = data
            inputs = inputs.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.float)
            outputs = net(inputs)
            loss = heatmap_loss(outputs, labels)
            running_loss += loss.item()

            # for hm in outputs:
            #     kps_from_hm = decode_heatmap(hm)

    running_loss /= len(loader.dataset)
    wandb.log({'val_loss': running_loss, 'epoch': epoch})
    print("Testing loss:", running_loss, end="\n-----------------------------------------------------------\n\n")
    return running_loss


def wandb_config(args):
    wandb.config.split = args.split
    wandb.config.seed = args.seed
    wandb.config.batch_size = args.batch_size
    wandb.config.epochs = args.epochs
    wandb.config.radius = args.radius
    wandb.config.learning_rate = args.lr


def main(args):
    create_folder(args.saved_folder)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create network
    dims = [[256, 256, 384], [384, 384, 512]]
    net = StackedHourglass(3, dims, 15).to(device)

    keypoint_label_names = list(range(15))
    dataset = KeypointDataset(args.annotation,
                              args.images,
                              keypoint_label_names=keypoint_label_names,
                              downsampling_factor=4,
                              in_memory=args.in_memory,
                              radius=args.radius)

    num_training = int(len(dataset) * args.split)
    num_testing = len(dataset) - num_training
    training_set, test_set = random_split(dataset, [num_training, num_testing],
                                          generator=torch.Generator().manual_seed(args.seed))

    train_loader = torch.utils.data.DataLoader(training_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size * 2, drop_last=False)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)

    curr_train_loss = 2.0
    curr_valid_loss = 10.0

    wandb.init(project="facial-landmark")

    for epoch in range(1, args.epochs+1):  # loop over the dataset multiple times
        print("Training epoch", epoch)
        train_loss = train_one_epoch(net, optimizer, train_loader, epoch, device)
        val_loss = run_validation(net, test_loader, epoch, device)
        if train_loss < curr_train_loss:
            torch.save(net.state_dict(), "{}/HG_best_train.pt".format(args.saved_folder))
            curr_train_loss = train_loss

        if val_loss < curr_valid_loss:
            torch.save(net.state_dict(), "{}/HG_best_val.pt".format(args.saved_folder))
            curr_valid_loss = val_loss

    torch.save(net, "{}/HG_final.pt".format(args.saved_folder))
    print('Finished Training')
    return net


if __name__ == '__main__':
    args = parse_args()
    main(args)
