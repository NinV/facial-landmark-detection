import json
import pathlib

from easydict import EasyDict as edict
from tqdm import tqdm
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .heatmap import heatmap_from_objects, heatmap_from_kps
from ..utils.image import load_image, per_image_normalization, letterbox, simple_normalize


class BaseDataset(Dataset):
    def __init__(self, annotation_file, image_folder, in_memory=False,
                 downsampling_factor=4, image_size=(512, 512), normalize_func=simple_normalize,
                 augmentation=None, keypoint_label_names=None):
        super(BaseDataset, self).__init__()
        self.annotation_file = annotation_file
        self.image_folder = pathlib.Path(image_folder)
        self.in_memory = in_memory
        self.augmentation = augmentation
        self.downsampling_factor = downsampling_factor
        self.normalize_func = normalize_func
        self.preprocess_func = Resize(image_size)

        # using dictionary instead of list for cases where some image files cannot be read
        self.images = {}
        self._image_ids = []    # for holding image_id(s) of valid image_files
        self._load_images()

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

    def __len__(self):
        return len(self._image_ids)
        # return 10

# class DetectionDataset(BaseDataset):
#     def __init__(self, *args, **kwargs):
#         super(DetectionDataset, self).__init__(*args, **kwargs)
#         self.annotations = {}
#         self.object_label_names = {}
#
#         self._parse_label_file()
#
#     def _parse_label_file(self):
#         """
#         for COCO format
#         """
#         for ann in self.annotation_file_data.annotations:
#             if ann.image_id in self.annotations:
#                 self.annotations[ann.image_id].append([ann.bbox] + [ann.category_id])
#             else:
#                 self.annotations[ann.image_id] = [[ann.bbox] + [ann.category_id]]
#
#         for cat in self.annotation_file_data.categories:
#             self.object_label_names[cat.id] = cat.name
#
#         self._num_classes = len(self.object_label_names)
#
#     def __getitem__(self, idx):
#         idx = self._image_ids[idx]
#
#         if not self.in_memory:
#             X = load_image(self.images[idx])
#         else:
#             X = self.images[idx]
#         X = self.normalize_func(X)
#         Y = self.annotations[idx]
#         if self.augmentation is not None:
#             X, Y = self.augmentation.transform(X, Y)
#         h, w, c = X.shape
#         Y = heatmap_from_objects((h, w, self._num_classes), Y)
#         X = torch.from_numpy(X).permute(2, 0, 1).to(self.device)
#         return X, Y


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

    def _parse_label_file(self):
        """
        for COCO format
        """
        for ann in self.annotation_file_data.annotations:
            if ann.image_id in self.annotations:
                self.annotations[ann.image_id].append([ann.bbox + [ann.category_id]] + [ann.keypoints])
            else:
                self.annotations[ann.image_id] = [[ann.bbox + [ann.category_id]] + [ann.keypoints]]

        # for cat in data.categories:
            # self.object_label_names[cat.id] = cat.name
            # self.keypoint_label_names[cat.]
        self._num_classes = len(self.keypoint_label_names)

    def __getitem__(self, idx):
        idx = self._image_ids[idx]

        if not self.in_memory:
            X = load_image(self.images[idx])
        else:
            X = self.images[idx]

        Y = self._process_kps_annotation(self.annotations[idx])

        if self.preprocess_func is not None:
            X, Y = self.preprocess_func(X, Y)

        if self.normalize_func is not None:
            X = self.normalize_func(X)

        if self.augmentation is not None:
            X, Y = self.augmentation.transform(X, Y)
        h, w, c = X.shape
        Y = heatmap_from_kps((h//self.downsampling_factor, w//self.downsampling_factor, self._num_classes),
                             Y, radius=self.radius)

        X = torch.from_numpy(X).permute(2, 0, 1)
        Y = torch.from_numpy(Y).permute(2, 0, 1)
        return X, Y

    def _process_kps_annotation(self, labels, visible_only=True):
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
                kps_.append([x//self.downsampling_factor, y//self.downsampling_factor, i])
        return kps_


class Resize:
    def __init__(self, image_size, training=True):
        self.w, self.h = image_size
        self.training = training

    def __call__(self, img, kps):
        resized_img, ratio, (dw, dh) = letterbox(img, new_shape=(self.h, self.w), auto=not self.training)
        kps_loc = np.asarray(kps, dtype=np.float)
        kps_loc[:, :2] *= ratio
        kps_loc[:, 0] += dw
        kps_loc[:, 1] += dh
        return resized_img, kps


