import random
import numpy as np
import cv2


class SequentialTransform:
    def __init__(self, geometric_transforms, geometric_transform_prob,
                 color_distortions, color_distortions_prob, out_size,
                 shuffle=True, color_mode='bgr', interpolation=cv2.INTER_AREA,
                 border_mode=cv2.BORDER_CONSTANT, border_value=(114, 114, 114)):
        self.geometric_transforms = geometric_transforms
        self.geometric_transform_prob = geometric_transform_prob
        self.color_distortions = color_distortions
        self.color_distortions_prob = color_distortions_prob
        self.out_size = out_size

        self.shuffle = shuffle
        self.color_mode = color_mode
        self.interpolation = interpolation
        self.border_mode = border_mode
        self.border_value = border_value

    def _get_transformation_matrix(self, img_size):
        if self.shuffle:
            temp = list(zip(self.geometric_transforms, self.geometric_transform_prob))
            random.shuffle(temp)
            self.geometric_transforms, self.geometric_transform_prob = zip(*temp)

        w, h = img_size
        T = np.identity(3)
        for transform, prob in zip(self.geometric_transforms, self.geometric_transform_prob):

            if random.random() < prob:
                T = np.matmul(transform.get_transformation_matrix((w, h)), T)
        return T

    def transform(self, image: np.ndarray, points=None):
        """
        :param image: numpy array
        :param points: [[x1, y1], [x2, y2], ...,[xn, yn]].
        :return:
        """
        h, w = image.shape[:2]
        T = self._get_transformation_matrix(img_size=(w, h))
        out = cv2.warpPerspective(image.copy(), T, self.out_size, None,
                                  self.interpolation, self.border_mode, self.border_value)
        if points is not None:
            points = np.array(points, dtype=np.float)

            # convert to homogeneous coordinates
            if points.shape[1] == 2:
                nums = points.shape[0]
                points = np.hstack((points, np.ones((nums, 1), dtype=np.float)))
            points = np.matmul(T, points.T).T
            points = points[:, :2]

        for color_distortion, prob in zip(self.color_distortions, self.color_distortions_prob):
            if random.random() < prob:
                out = color_distortion.random_distort(out, self.color_mode)
        return out, points


class ColorDistortion:
    def __init__(self, hue=0.2, saturation=1.5, exposure=1.5):
        self.hue = hue
        self.saturation = saturation
        self.exposure = exposure

    def random_distort(self, image, mode="bgr"):
        if mode == "bgr":
            flag_to_hsv = cv2.COLOR_BGR2HSV
            flag_from_hsv = cv2.COLOR_HSV2BGR
        elif mode == "rgb":
            flag_to_hsv = cv2.COLOR_RGB2HSV
            flag_from_hsv = cv2.COLOR_HSV2RGB
        else:
            raise ValueError("unrecognised color mode {}".format(mode))
        dhue = np.random.uniform(-self.hue, self.hue)
        dsat = self._rand_scale(self.saturation)
        dval = self._rand_scale(self.exposure)
        image_hsv = cv2.cvtColor(image, flag_to_hsv)
        image_hsv[:, :, 1] = cv2.multiply(image_hsv[:, :, 1], dsat)
        image_hsv[:, :, 2] = cv2.multiply(image_hsv[:, :, 2], dval)
        image_hsv = cv2.add(image_hsv, dhue)
        return cv2.cvtColor(image_hsv, flag_from_hsv)

    @staticmethod
    def _rand_scale(s):
        scale = np.random.uniform(1, s)
        if np.random.uniform(0, 1) < 0.5:
            return scale
        return 1 / scale


class GaussianBlur:
    def __init__(self, prob=0.5, ksize=(5, 5)):
        self.prob = prob
        self.ksize = ksize

    def random_distort(self, image, mode="bgr"):
        if random.random() < self.prob:
            return cv2.GaussianBlur(image, self.ksize, 0)
        return image


class RandomTranslation:
    def __init__(self, tx_range, ty_range):
        self._validate_input(tx_range, ty_range)
        self.tx_range = tx_range
        self.ty_range = ty_range

    def _validate_input(self, *args):
        for arg in args:
            if len(arg) != 2:
                raise ValueError("Both tx_range and ty_range must have length of 2")
            min_value = min(arg)
            max_value = max(arg)
            if min_value < -1.:
                raise ValueError("translation range must not < -1")

            if max_value > 1.:
                raise ValueError("translation range must not > 1")

    def get_transformation_matrix(self, img_size):
        iw, ih = img_size
        tx = random.uniform(*self.tx_range) * iw
        ty = random.uniform(*self.tx_range) * ih

        T = np.array([[1, 0, tx],
                      [0, 1, ty],
                      [0, 0, 1]])
        return T


class RandomScalingAndRotation:
    def __init__(self, angle_range, scale_range):
        """
        :param angle_range: angle range in degree
        :param scale_range: scale range
        :param center: center point. Default: (0, 0)
        """
        self._validate_input(angle_range, scale_range)
        self.angle_range = angle_range
        self.scale_range = scale_range

    def _validate_input(self, *args):
        angle_range, scale_range = args
        if len(angle_range) != 2:
            raise ValueError("angle_range must have length of 2")

        if len(scale_range) != 2:
            raise ValueError("scale_range must have length of 2")

        for value in scale_range:
            if value < 0:
                raise ValueError("scale_range must not < 0")

    def get_transformation_matrix(self, img_size):
        iw, ih = img_size
        center = iw/2, ih/2
        angle = random.uniform(*self.angle_range)
        scale = random.uniform(*self.scale_range)

        T = cv2.getRotationMatrix2D(center, angle, scale)
        T = np.vstack((T, np.array([[0, 0, 1]])))
        return T
