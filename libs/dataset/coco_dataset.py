import json
import pathlib

from easydict import EasyDict as edict
from tqdm import tqdm
import numpy as np
import torch

from .dataset import BaseDataset
from libs.utils.image import load_image, Resize
from libs.utils.heatmap import heatmap_from_kps


class KeypointDataset(BaseDataset):
    def __init__(self, *args, crop_from_boxes=False, radius=4, **kwargs):
        super(KeypointDataset, self).__init__(*args, **kwargs)
        self.crop_from_boxes = crop_from_boxes

        self.annotations = {}
        self.object_label_names = {}
        self.keypoint_label_names = kwargs["keypoint_label_names"]
        self._num_classes = len(self.keypoint_label_names)
        self.radius = radius

        self._parse_label_file()

    def _load_images(self):
        with open(self.annotation_file, "r") as f:
            data = edict(json.load(f))
        self.annotation_file_data = data

        print("Checking image files")
        for img_data in tqdm(data.images):
            try:
                image_file_path = self.image_folder / img_data.file_name
                img = load_image(image_file_path)
            except FileNotFoundError:
                continue

            if self.in_memory:
                self.images[img_data.id] = img
            else:
                self.images[img_data.id] = image_file_path
            self._image_ids.append(img_data.id)

    def _parse_label_file(self):
        """
        for COCO format
        """
        for ann in self.annotation_file_data.annotations:
            if ann.image_id in self.annotations:
                self.annotations[ann.image_id].append([ann.bbox + [ann.category_id]] + [ann.keypoints])
            else:
                self.annotations[ann.image_id] = [[ann.bbox + [ann.category_id]] + [ann.keypoints]]

        self._num_classes = len(self.keypoint_label_names)

    def __getitem__(self, idx):
        transform_params = None
        idx = self._image_ids[idx]

        if not self.in_memory:
            X = load_image(self.images[idx])
        else:
            X = self.images[idx]

        Y = self._process_kps_annotation(self.annotations[idx])

        if self.resize_func is not None:
            X, Y, transform_params = self.resize_func(X, Y)  # resize image

        if self.normalize_func is not None:
            X = self.normalize_func(X)

        if self.augmentation is not None:
            X, Y = self.augmentation.transform(X, Y)

        h, w, c = X.shape
        hm = heatmap_from_kps((h // self.downsampling_factor, w // self.downsampling_factor, self._num_classes),
                              self._downsample_heatmap_kps(Y), radius=self.radius)

        X = torch.from_numpy(X).permute(2, 0, 1)
        hm = torch.from_numpy(hm).permute(2, 0, 1)

        # normalize keypoint location
        Y[:, 0] /= w
        Y[:, 1] /= h
        if transform_params is None:
            transform_params = torch.tensor([])
        else:
            transform_params = torch.tensor(transform_params)
        return X, Y[:, :2], hm, transform_params

    @staticmethod
    def _process_kps_annotation(labels, visible_only=True):
        """
        return: kps_ [[x1, y1, classId], ...]
        """
        kps = [l[1] for l in labels]
        kps_ = []
        for object_kps in kps:
            object_kps = np.asarray(object_kps).reshape(-1, 3)
            for i, (x, y, v) in enumerate(object_kps):
                if visible_only and v != 2:
                    continue
                kps_.append([x, y, i])
        return kps_

    def _downsample_heatmap_kps(self, kps):
        kps[:, :2] /= self.downsampling_factor
        return kps
