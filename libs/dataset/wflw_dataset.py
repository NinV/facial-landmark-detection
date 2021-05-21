import pathlib

from tqdm import tqdm
import pandas as pd
import numpy as np
import cv2
import torch

from .dataset import BaseDataset
from libs.utils.image import load_image, Resize
from libs.utils.heatmap import heatmap_from_kps


class WFLWDataset(BaseDataset):
    def __init__(self, *args, radius=4, crop_face_storing="temp", **kwargs):
        self.keypoint_label_names = kwargs["keypoint_label_names"]
        self._num_classes = len(self.keypoint_label_names)
        self.radius = radius
        self.crop_face_storing = crop_face_storing
        super(WFLWDataset, self).__init__(*args, **kwargs)

    def _load_images(self):
        """
        coordinates of 98 landmarks (196) + coordinates of upper left corner and lower right corner of detection
        rectangle (4) + attributes annotations (6) + image name
        (196) x0 y0 ... x97 y97
        (4) x_min_rect y_min_rect x_max_rect y_max_rect
        (6) pose expression illumination make-up occlusion blur
        (1) image_name
        """
        if not self.in_memory:
            self.temp_images_folder = pathlib.Path(self.crop_face_storing)
            self.temp_images_folder.mkdir(parents=True, exist_ok=True)

        csv_headers = []
        for i in range(self._num_classes):
            csv_headers.extend(("x{}".format(i), "y{}".format(i)))
        # csv_headers = [("x{}".format(i), "y{}".format(i)) for i in range(98)]
        csv_headers.extend(("x_min_rect", "y_min_rect", "x_max_rect", "y_max_rect"))
        csv_headers.extend(("pose", "expression", "illumination", "make-up", "occlusion", "blur"))
        csv_headers.append("image_name")

        df = pd.read_csv(self.annotation_file, names=csv_headers, sep=" ")

        print("Loading dataset")
        self.annotations = {}
        for i, (_, row) in tqdm(enumerate(df.iterrows()), total=len(df.index)):
            img = load_image(self.image_folder / row.image_name)

            # may remove some key points
            crop = img[row.y_min_rect: row.y_max_rect,
                       row.x_min_rect: row.x_max_rect]

            lm_data = np.array(row.to_list()[:self._num_classes*2]).reshape(-1, 2)
            lm_data -= (row.x_min_rect, row.y_min_rect)
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
            img = self.images[idx]

        kps = self.annotations[idx].copy()  # always using a deep copy to prevent modification on original data
        if self.resize_func is not None:
            img, kps, transform_params = self.resize_func(img, kps)  # resize image

        if self.normalize_func is not None:
            img = self.normalize_func(img)

        if self.augmentation is not None:
            img, kps = self.augmentation.transform(img, kps)

        h, w, c = img.shape
        hm = heatmap_from_kps((h // self.downsampling_factor, w // self.downsampling_factor, self._num_classes),
                              self._downsample_heatmap_kps(kps), radius=self.radius)

        img = torch.from_numpy(img).permute(2, 0, 1)
        hm = torch.from_numpy(hm).permute(2, 0, 1)

        # normalize keypoint location
        # kps[:, 0] /= w
        # kps[:, 1] /= h

        if transform_params is None:
            transform_params = torch.tensor([])
        else:
            transform_params = torch.tensor(transform_params)
        return img, kps, hm, transform_params

    def _downsample_heatmap_kps(self, kps):
        kps[:, :2] /= self.downsampling_factor
        return kps

