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
import h5py
import scipy.io as sio
class COFWDataset(BaseDataset):
    def __init__(self, *args,radius=4, crop_face_storing='tem/test', hrnet_box=False, **kwargs):
        self.keypoint_label_names = kwargs["keypoint_label_names"]
        self._num_classes = len(self.keypoint_label_names)
        self.radius = radius
        self.hrnet_box = hrnet_box
        # self.annotation_files = pathlib.Path(self.annotation_file)
        self.crop_face_storing = crop_face_storing
        
        super(COFWDataset, self).__init__(*args, **kwargs)
        self._image_ids = [i for i in range(len(self.annotations))]
    def _load_images(self):
        
        self.matlab_image_file = self.annotation_file
        if not self.in_memory:
            self.temp_images_folder = pathlib.Path(self.crop_face_storing)
            self.temp_images_folder.mkdir(parents=True, exist_ok=True)
        #self.images is path to matlab file
        mat = h5py.File(self.matlab_image_file, 'r')
        print(mat)
        imgs = mat['IsT']
        pts = mat['phisT']
        bboxes = mat['bboxesT']
        num = imgs.shape[1]
        self.annotations = {}
        print("Loading dataset")
        for idx in tqdm(range(num), total=num):
            name =  str(idx+1) + '_points.mat'
            file =  os.path.join(self.image_folder,name)
            keypoints = sio.loadmat(file)['Points']
            img = np.array(mat[imgs[0, idx]]).transpose()
            try :
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            except:
                print('error convert image color',name)
                continue
            h, w = img.shape[:2]
            points = np.array(keypoints).reshape(-1,2)
            if self.hrnet_box:
                lm_data = np.array(points.flatten()[:self._num_classes*2]).reshape(-1, 2)
                x_min, x_max = math.floor(np.min(lm_data[:, 0])), math.ceil(np.max(lm_data[:, 0]))
                y_min, y_max = math.floor(np.min(lm_data[:, 1])), math.ceil(np.max(lm_data[:, 1]))
                x_min, x_max = max(0, x_min), min(w-1, x_max)
                y_min, y_max = max(0, y_min), min(h - 1, y_max)
                crop = img[y_min: y_max,
                           x_min: x_max]
                lm_data -= (x_min, y_min)
            else :
                print('Need hrnet_box is True')
            kp_classes = np.arange(self._num_classes).reshape(self._num_classes, 1)
            lm_data = np.concatenate([lm_data, kp_classes], axis=-1)
            save_path = str(self.temp_images_folder / "{}.png".format(str(idx+1)))
            self.images[idx] = save_path
            self._image_ids.append(idx+1)
            self.annotations[idx] = lm_data
            cv2.imwrite(save_path, crop)
            
    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError
        img = load_image(self.images[idx]).copy()

        kps_tmp = self.annotations[idx]  # always using a deep copy to prevent modification on original data
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

