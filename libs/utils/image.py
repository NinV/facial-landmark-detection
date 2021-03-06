import random
import cv2
import numpy as np


def load_image(image_file):
    """
    :param image_file: str or Path like object
    """
    img = cv2.imread(str(image_file))
    if img is None:
        raise FileNotFoundError("cannot read image file at:", image_file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


class Resize:
    def __init__(self, image_size, training=True):
        self.w, self.h = image_size
        self.training = training

    def __call__(self, img, kps):
        resized_img, ratio, (dw, dh) = letterbox(img, new_shape=(self.h, self.w), auto=not self.training)
        kps_resized = np.asarray(kps, dtype=np.float)
        kps_resized[:, :2] *= ratio  # ratio = [ratio_w, ratio_h]
        kps_resized[:, 0] += dw
        kps_resized[:, 1] += dh

        return resized_img, kps_resized, [*ratio, dw, dh]

    @staticmethod
    def inverse_resize(kps, ratio, dw, dh):
        if not isinstance(kps, np.ndarray):
            kps_ = np.asarray(kps, dtype=np.float)
        else:
            kps_ = kps.copy()
        kps_[:, 0] -= dw
        kps_[:, 1] -= dh
        kps_[:, :2] /= ratio
        return kps_


def per_image_normalization(image):
    image = image.astype(np.float)
    try:
        c = image.shape[2]
        for c_ in range(c):
            image[:, :, c_] -= np.mean(image[:, :, c_])
            image[:, :, c_] /= np.std(image[:, :, c_])
    except IndexError:
        image[:, :] -= np.mean(image[:, :])
        image[:, :] /= np.std(image[:, :])
    return image


def simple_normalize(image):
    return image / 255.


def mean_std_normalize(image, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    image = image / 255.
    image -= mean
    image /= std
    return image


def reverse_mean_std_normalize(image, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    image *= std
    image += mean
    image *= 255
    return image
