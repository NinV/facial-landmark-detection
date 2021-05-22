import argparse
import numpy as np
import cv2
import torch


from libs.models.networks.models import HGLandmarkModel
from libs.dataset.wflw_dataset import WFLWDataset
from libs.models.losses import heatmap_loss


def parse_args():
    parser = argparse.ArgumentParser()

    # model config
    parser.add_argument("--num_classes", default=98, type=int, help="Number of landmark classes")
    parser.add_argument("--image_size", default=512, type=int)
    parser.add_argument("--downsample", action="store_false", help="Disable downsampling")

    # dataset
    parser.add_argument("-i", "--images", required=True, help="Path to image folder for training")
    parser.add_argument("--annotation", required=True, help="Annotation file for training")
    parser.add_argument("--format", default="WFLW", help="dataset format: 'WFLW', 'COCO'")
    parser.add_argument("--in_memory", action="store_true", help="Load all image on RAM")
    # parser.add_argument("-n", "--normalized_index", nargs='+', type=int, default=[96, 97],
    #                     help="landmarks indexes for calculate normalize distance in NME")
    parser.add_argument("--radius", type=int, default=8)

    # save config
    parser.add_argument("-s", "--saved_weights", default="saved_models/cnn-baseline-1/HG_epoch_47.pt",
                        help="path to saved weights")
    parser.add_argument("--batch_size", type=int, default=4)
    return parser.parse_args()


def plot_kps(img, gt, pred):
    for (x, y, classId) in gt:
        cv2.circle(img, (int(x + 0.5), int(y + 0.5)), radius=2, thickness=-1, color=[0, 255, 0])

    for (x, y, classId) in pred:
        cv2.circle(img, (int(x + 0.5), int(y + 0.5)), radius=2, thickness=-1, color=[0, 0, 255])

    return img


def visualize_hm(hm):
    hm = hm.permute(1, 2, 0).cpu().numpy()
    hm = np.max(hm, axis=-1)
    return (hm * 255).astype(np.uint8)


def run_evaluation(net, dataset, device):
    net.eval()
    running_hm_loss = 0
    with torch.no_grad():
        for i, data in enumerate(dataset):
            img, gt_kps, gt_hm, _ = data
            img_tensor = torch.unsqueeze(img, 0).to(device, dtype=torch.float)
            gt_kps = np.expand_dims(gt_kps, axis=0) * net.downsampling_factor

            pred_hm_tensor = net(img_tensor)
            pred_kps = net.decode_heatmap(pred_hm_tensor, confidence_threshold=0.0) * net.downsampling_factor
            pred_kps = pred_kps.detach().cpu().numpy()

            # show image
            img = img * 255
            img = img.detach().cpu()
            img = img.permute(1, 2, 0).numpy().astype(np.uint8)
            img = plot_kps(img, gt_kps[0], pred_kps[0])

            gt_hm_np = visualize_hm(gt_hm)
            pred_hm_np = visualize_hm(pred_hm_tensor[0])

            cv2.imshow("img", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            cv2.imshow("gt_hm", gt_hm_np)
            cv2.imshow("pred_hm", pred_hm_np)
            k = cv2.waitKey(0)

            if k == ord("q"):
                break


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create network
    dims = [[256, 256, 384], [384, 384, 512]]

    graph_model_configs = None
    net = HGLandmarkModel(3, args.num_classes, dims, graph_model_configs, device,
                          include_graph_model=False, downsample=args.downsample)
    net.load_state_dict(torch.load(args.saved_weights))
    net.eval()
    keypoint_label_names = list(range(args.num_classes))
    dataset = WFLWDataset(args.annotation,
                          args.images,
                          image_size=(args.image_size, args.image_size),
                          keypoint_label_names=keypoint_label_names,
                          downsampling_factor=net.downsampling_factor,
                          in_memory=args.in_memory,
                          crop_face_storing="temp/train",
                          radius=args.radius,
                          )
    run_evaluation(net, dataset, device)


if __name__ == '__main__':
    args = parse_args()
    main(args)
