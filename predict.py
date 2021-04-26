import os
import json
from tqdm import tqdm
import pathlib
import cv2
import numpy as np
import torch
from libs.dataset.dataset import simple_normalize, letterbox
from libs.models.networks.hourglass import StackedHourglass
from libs.utils.image import load_image


def decode_heatmap(hm, kernel=3, num_classes=15, conf=0.5):
    """
    hm : (h, w, c) numpy array
    """
    pad = (kernel - 1) // 2
    hmax = torch.nn.functional.max_pool2d(hm, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == hm).float()
    hm_reduce = (hm * keep)[0]

    # find indices
    kps = {}
    for c in range(num_classes):
        indices_y, indices_x = torch.logical_and(hm_reduce[c] > conf,
                                                 hm_reduce[c] == torch.max(hm_reduce[c])).nonzero(as_tuple=True)

        indices_x = indices_x.tolist()
        indices_y = indices_y.tolist()
        if indices_x:
            kps[c] = (indices_x[0], indices_y[0])
        # print(indices_x, indices_y)
        # for x, y in zip(indices_x, indices_y):
        #     try:
        #         kps[c].append((x, y))
        #     except KeyError:
        #         kps[c] = [(x, y)]

    # for plotting
    hm_reduce = hm_reduce.permute(1, 2, 0)
    return hm_reduce.detach().cpu().numpy(), kps


def preproc(img):
    resized, ratio, (dw, dh) = letterbox(img, new_shape=(512, 512), auto=True)
    img = resized/255
    img = np.expand_dims(img, axis=0)
    img = torch.from_numpy(img).permute(0, 3, 1, 2).float()
    return img, resized


def main():
    saved_model_path = "saved_models/HG_best_train.pt"
    # img_path = "/home/ninv/MyProjects/deneer/side-face-data-3-3/Folder02/success/1 (3) (Large).jpg"
    # img_path = "/media/ninv/Data/dataset/deneer/side-face-data-9-4/Folder34/success/koder (42).jpg"
    img_folder = pathlib.Path("/media/ninv/Data/dataset/deneer/side-face-data-9-4/Folder34/success")

    dims = [[256, 256, 384], [384, 384, 512]]
    net = StackedHourglass(3, dims, 15)
    net.load_state_dict(torch.load(saved_model_path))
    net.eval()

    img_files = list(img_folder.glob("*.jpg"))
    for i, img_path in enumerate(tqdm(img_files)):
        with torch.no_grad():
            img = load_image(img_path)
            img_preproc, resized = preproc(img)
            pred = net(img_preproc)
            pred, kps_on_heatmap = decode_heatmap(pred, conf=0.1)
        for kp_name, (x, y) in kps_on_heatmap.items():
            cv2.circle(resized, (x * net.downsampling_factor, y*net.downsampling_factor), radius=2,
                       thickness=-1, color=[0, 255, 0])

        # pred = np.max(pred, axis=2, keepdims=False) * 255
        # cv2.imshow("image", cv2.cvtColor(resized, cv2.COLOR_RGB2BGR))
        # cv2.imshow("pred", pred.astype(np.uint8))
        # k = cv2.waitKey(0)
        # if k == ord("q"):
        #     break
        cv2.imwrite("out/img_{}.jpg".format(i+1), cv2.cvtColor(resized, cv2.COLOR_RGB2BGR))
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
