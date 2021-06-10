import pathlib
import math
import json
from tqdm import tqdm
import pandas as pd
import numpy as np
import cv2
import torch
import os
from .dataset import BaseDataset
from libs.utils.image import load_image, Resize
from libs.utils.heatmap import heatmap_from_kps
from libs.utils.image import mean_std_normalize
import torch
class COFWDataset(BaseDataset):
    def __init__(self, *args, radius=4, crop_face_storing="temp", hrnet_box=False, **kwargs):
        super(COFWDataset, self).__init__(*args, **kwargs)
        self.keypoint_label_names = kwargs["keypoint_label_names"]
        self._num_classes = len(self.keypoint_label_names)
        self.radius = radius


        # load json
        f = open(self.annotation_file)
        data = json.load(f)
        self.images= data['images']
        self.annotations = data["annotations"]
        self._image_ids = [i for i in range(len(self.images))]
        
    def _load_images(self):
        return 
            
    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError
        img = load_image(os.path.join(self.image_folder,self.images[idx]['file_name'])).copy()

        kps_tmp = np.array(self.annotations[idx]['keypoints']).reshape(-1,3)[:,:2]  # always using a deep copy to prevent modification on original data
        kp_classes = np.arange(self._num_classes).reshape(self._num_classes, 1)
        kps = np.concatenate([kps_tmp, kp_classes], axis=-1)
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

