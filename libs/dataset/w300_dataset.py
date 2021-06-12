import pathlib
import math

from tqdm import tqdm
import pandas as pd
import numpy as np
import cv2
import torch
import glob
from .dataset import BaseDataset
from libs.utils.image import load_image, Resize
from libs.utils.heatmap import heatmap_from_kps


class W300_Dataset(BaseDataset):
    def __init__(self, *args, radius=4, crop_face_storing="temp", hrnet_box=False, **kwargs):    
        self.keypoint_label_names = kwargs["keypoint_label_names"]
        self._num_classes = len(self.keypoint_label_names)
        self.radius = radius
        self.crop_face_storing = crop_face_storing
        self.hrnet_box = hrnet_box
        super(W300_Dataset, self).__init__(*args, **kwargs) 
        



    def _load_images(self):
        """
        coordinates of 68 landmarks (68) + coordinates of upper left corner and lower right corner of detection
        rectangle (4) + attributes annotations (6) + image name
        (196) x0 y0 ... x97 y97
        (6) pose expression illumination make-up occlusion blur
        (1) image_name
        """
        self.annotation_files = glob.glob(str(self.image_folder)+'/*/*.pts')
        if not self.in_memory:
            self.temp_images_folder = pathlib.Path(self.crop_face_storing)
            self.temp_images_folder.mkdir(parents=True, exist_ok=True)

        csv_headers = []
        for i in range(self._num_classes):
            csv_headers.extend(("x{}".format(i), "y{}".format(i)))
        # csv_headers = [("x{}".format(i), "y{}".format(i)) for i in range(98)]
        # csv_headers.extend(("x_min_rect", "y_min_rect", "x_max_rect", "y_max_rect"))
        # csv_headers.extend(("pose", "expression", "illumination", "make-up", "occlusion", "blur"))
        csv_headers.append("image_name")

        # df = pd.read_csv(self.annotation_file, names=csv_headers, sep=" ")

        print("Loading dataset")
        self.annotations = {}
        for i, (filename) in tqdm(enumerate(self.annotation_files),total=len(self.annotation_files)):
            img = load_image(filename.replace('pts','png'))
            h, w = img.shape[:2]
            _keypoints = np.loadtxt(filename, comments=("version:", "n_points:", "{", "}"))

            if self.hrnet_box:
                lm_data = np.array(_keypoints.flatten()[:self._num_classes*2]).reshape(-1, 2)
                x_min, x_max = math.floor(np.min(lm_data[:, 0])), math.ceil(np.max(lm_data[:, 0]))
                y_min, y_max = math.floor(np.min(lm_data[:, 1])), math.ceil(np.max(lm_data[:, 1]))
                x_min, x_max = max(0, x_min), min(w-1, x_max)
                y_min, y_max = max(0, y_min), min(h - 1, y_max)
                crop = img[y_min: y_max,
                           x_min: x_max]
                lm_data -= (x_min, y_min)
            else:
                print('Need hrnet_box is True')
                # may remove some key points
                # crop = img[row.y_min_rect: row.y_max_rect,
                #            row.x_min_rect: row.x_max_rect]
                # lm_data = np.array(row.to_list()[:self._num_classes * 2]).reshape(-1, 2)
                # lm_data -= (row.x_min_rect, row.y_min_rect)

            kp_classes = np.arange(self._num_classes).reshape(self._num_classes, 1)
            lm_data = np.concatenate([lm_data, kp_classes], axis=-1)
            self.annotations[i] = lm_data
            if self.in_memory:
                self.images[i] = crop
                self._image_ids.append(i)
            else:
                save_path = str(self.temp_images_folder / "{}.png".format(i))
                cv2.imwrite(save_path, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
                self.images[i] = save_path
                self._image_ids.append(i)

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError

        if not self.in_memory:
            img = load_image(self.images[idx])
        else:
            img = self.images[idx].copy()

        kps = self.annotations[idx].copy()  # always using a deep copy to prevent modification on original data
        if self.resize_func is not None:
            img, kps, resize_params = self.resize_func(img, kps)  # resize image
            resize_params = torch.tensor(resize_params)
        else:
            resize_params = torch.tensor([])

        if self.augmentation is not None:
            img, kps_ = self.augmentation.transform(img, kps[:,:2])
            kps[:,:2] = kps_

        if self.normalize_func is not None:
            img = self.normalize_func(img)

        h, w, c = img.shape
        hm = heatmap_from_kps((h // self.downsampling_factor, w // self.downsampling_factor, self._num_classes),
                              self._downsample_heatmap_kps(kps), radius=self.radius)

        img = torch.from_numpy(img).permute(2, 0, 1)
        hm = torch.from_numpy(hm).permute(2, 0, 1)

        # normalize keypoint location
        # kps[:, 0] /= w
        # kps[:, 1] /= h
        return img, kps, hm, resize_params

    def _downsample_heatmap_kps(self, kps):
        kps[:, :2] /= self.downsampling_factor
        return kps

