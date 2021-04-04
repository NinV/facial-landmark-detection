import argparse
import pathlib

import torch

from libs.models.networks.hourglass import StackedHourglass
from libs.dataset.dataset import KeypointDataset
from libs.models.losses import heatmap_loss


def parse_args():
    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument("-i", "--images", required=True, help="Path to image folder")
    parser.add_argument("--annotation", required=True, help="Annotation file (.json)")
    parser.add_argument("--in_memory", action="store_true", help="Load all image on RAM")

    # save config
    parser.add_argument("-s", "--saved_folder", default="saved_models", help="folder for saving model")
    parser.add_argument("--save_best_only", action="store_true", help="only save best weight")
    parser.add_argument("--valid_interval", type=int, default=2, help="Save model and evaluate interval")

    # training hyperparameters
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    # parser.add_argument("--learning_rate", type=float, default=10e-3, help="Initial learning rate")
    # parser.add_argument("--decay_steps", type=float, default=10000, help="learning rate decay step")
    # parser.add_argument("--decay_rate", type=float, default=0.995, help="learning rate decay rate")
    # parser.add_argument("--staircase", action="store_true", help="learning rate decay on step (default: smooth)")
    return parser.parse_args()


def create_folder(path):
    path = pathlib.Path(path)
    path.mkdir(parents=True, exist_ok=True)


def train(args):
    create_folder(args.saved_folder)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dims = [[256, 256, 384], [384, 384, 512]]
    net = StackedHourglass(3, dims, 15).to(device)

    keypoint_label_names = list(range(15))
    dataset = KeypointDataset(args.annotation,
                              args.images,
                              keypoint_label_names=keypoint_label_names,
                              downsampling_factor=4,
                              in_memory=False)

    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    for epoch in range(args.epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        print("Training epoch", epoch+1)
        for i, data in enumerate(loader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.float)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = heatmap_loss(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print("Average loss (across all mini batches):", running_loss/(i+1))
        torch.save(net.state_dict(), "saved_models/HG_ep_{}.pt".format(epoch+1))

    torch.save(net, "saved_models/HG_final.pt")
    print('Finished Training')
    return net


if __name__ == '__main__':
    args = parse_args()
    train(args)

