import sys
import argparse
import pathlib
project_path = str(pathlib.Path(__file__).absolute().parents[1])
sys.path.insert(0, project_path)

import numpy as np
import cv2
import torch
from easydict import EasyDict as edict

from libs.models.networks.models import LandmarkModel
from libs.utils.image import mean_std_normalize, load_image, letterbox

heatmap_model_config = {"in_channels": 3,
                        "num_classes": 98,
                        "hg_dims": [[256, 256, 384], [384, 384, 512]],
                        "downsample": True
                        }

graph_model_config = {"num_classes": 98,
                      "embedding_hidden_sizes": [32],
                      "class_embedding_size": 1,
                      "edge_hidden_size": 4,
                      "visual_feature_dim": 270,  # HRNet
                      "visual_hidden_sizes": [512, 128, 32],
                      "visual_embedding_size": 8,
                      "GCN_dims": [64, 16],
                      "self_connection": False,
                      "graph_norm": "softmax"
                      }


def parse_args():
    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument("-i", "--image", required=True, help="Input image")
    parser.add_argument("--image_size", default=256, type=int)
    parser.add_argument("--show", action="store_true")

    # model config
    parser.add_argument("-w", "--weights", default="saved_models/graph_base_line/frosty-spaceship-175-epoch_19.pt",
                        help="path to saved weights")

    # show config
    parser.add_argument("--heatmap", action="store_true", help="show heatmap")
    parser.add_argument("--edge", action="store_true", help="show edge")
    parser.add_argument("--save_visualization", default="tmp_vis", help="save location for visualizations")
    return parser.parse_args()


def visualize_hm(hm, merge=True):
    hm = hm.permute(1, 2, 0).cpu().numpy()
    hm[hm < 0] = 0
    if merge:
        hm = np.max(hm, axis=-1)
    return (hm * 255).astype(np.uint8)


def process_image(img):
    img = mean_std_normalize(img)
    img_tensor = torch.from_numpy(img).permute(2, 0, 1)
    img_tensor = torch.unsqueeze(img_tensor, dim=0)
    return img_tensor.float()


def plot_kpp(img, kps, color=(0, 255, 0)):
    for (x, y) in kps:
        cv2.circle(img, (int(x + 0.5), int(y + 0.5)), radius=2, thickness=-1, color=color)
    return img


def plot_edge(img, p1, p2, weight):
    color = int(weight * 255)
    p1 = (int(p1[0] + 0.5), int(p1[1] + 0.5))
    p2 = (int(p2[0] + 0.5), int(p2[1] + 0.5))
    cv2.line(img, p1, p2, color=[color, color, color], thickness=1, lineType=cv2.LINE_AA)
    return img


def predict(net, args, device):
    img = load_image(args.image)
    img, _, _ = letterbox(img, args.image_size)
    img_tensor = process_image(img)
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        pred_hm_tensor, pred_kps_graph = net(img_tensor)

    pred_kps_graph = pred_kps_graph.cpu()
    batch_size, num_classes, h, w = pred_hm_tensor.size()
    hm_size = torch.tensor([h, w])
    pred_kps_graph *= (hm_size * net.hm_model.downsampling_factor)
    pred_kps_graph = torch.squeeze(pred_kps_graph, dim=0)

    # pred_kps_hm = net.hm_model.decode_heatmap(pred_hm_tensor,
    #                                           confidence_threshold=0.0) * net.hm_model.downsampling_factor
    # pred_kps_hm = pred_kps_hm.detach().cpu().numpy()

    # plot image
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = plot_kpp(img, pred_kps_graph)
    cv2.imshow("graph prediction", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # write image
    saved_folder = pathlib.Path(args.save_visualization)
    saved_folder.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(saved_folder / "prediction.png"), img)
    print("saved prediction image at: ", str(saved_folder / "prediction.png"))

    if args.edge:
        for i in range(net.gcn_model.num_classes):
            clone = img.copy()
            edge_value = net.gcn_model.edge_values[i]
            edge_value /= torch.max(edge_value)
            for j in range(net.gcn_model.num_classes):
                if j == i:
                    continue
                if edge_value[j] > 0.5:
                    clone = plot_edge(clone, pred_kps_graph[i], pred_kps_graph[j], edge_value[j])
            cv2.imwrite(str(saved_folder / "edge_to_kp_{}.png".format(i)), clone)
        print("saved edge visualization at: ", saved_folder)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create network
    net = LandmarkModel(heatmap_model_config, edict(graph_model_config), device, use_hrnet=True,
                        hrnet_config="face_alignment_wflw_hrnet_w18.yaml")
    if args.weights:
        print("Load pretrained weight at:", args.weights)
        net.load_state_dict(torch.load(args.weights))

    net.eval()
    predict(net, args, device)


if __name__ == '__main__':
    args = parse_args()
    main(args)
