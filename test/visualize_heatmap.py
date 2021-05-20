import sys
from pathlib import Path
project_path = str(Path(__file__).absolute().parents[1])
sys.path.insert(0, project_path)

import json
import os
import cv2
import numpy as np
import torch
from libs.utils.heatmap import heatmap_from_kps


def to_int(jsonfile, out):
    with open(jsonfile) as f:
        data = json.load(f)

    for face in data["images"]:
        for k,v in face["box"].items():
            if k != "part":
                face["box"][k] = int(v)

        for kp in face["box"]["part"]:
            for k, v in kp.items():
                kp[k] = int(v)

    with open(out, "w") as f:
        json.dump(data, f, indent=2)


def decode_heatmap(hm, kernel=3):
    """
    hm : (h, w, c) numpy array
    """
    pad = (kernel - 1) // 2

    hm = torch.from_numpy(hm)
    hm = hm.permute(2, 0, 1)
    hm = torch.unsqueeze(hm, dim=0)
    print(hm.shape)

    hmax = torch.nn.functional.max_pool2d(hm, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == hm).float()
    hm_reduce = (hm * keep)[0]
    print(hm_reduce.shape)
    hm_reduce = hm_reduce.permute(1, 2, 0)

    return hm_reduce.detach().cpu().numpy()


def main():
    with open("../sample_data/Folder1_fix.json", "r") as f:
        data = json.load(f)

    image_dir = "/home/ninv/Desktop/temp"
    DOWNSAMPLING = 4
    KeypointRadius = 4
    for imgdata in data["images"]:
        imagefile = os.path.join(image_dir, imgdata["file"])
        img = cv2.imread(imagefile)
        h, w = img.shape[:2]
        resized = cv2.resize(img, (round(w/DOWNSAMPLING), round(h/DOWNSAMPLING)))
        h, w = resized.shape[:2]

        kps = []
        for kp in imgdata["box"]["part"]:
            kp_x = round(kp["x"] / DOWNSAMPLING)
            kp_y = round(kp["y"] / DOWNSAMPLING)
            kps.append((kp_x, kp_y, 0))
        hm = heatmap_from_kps((h, w, 1), kps, radius=KeypointRadius)
        # hm_nms = decode_heatmap(torch.from_numpy(hm).float(), kernel=KeypointRadius*2+1)
        hm_nms = decode_heatmap(hm, kernel=3)

        cv2.imshow("img", resized)
        cv2.imshow("heatmap", (hm * 255).astype(np.uint8))
        cv2.imshow("heatmap reduce", (hm_nms * 255).astype(np.uint8))
        # cv2.imshow("face", img[faceROI_y: faceROI_y + faceROI_h, faceROI_x: faceROI_x + faceROI_w])
        k = cv2.waitKey(0)
        if k == ord("q"):
            break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
