import random
import numpy as np
import cv2


MATCHED_PARTS = {
    "300W": ([1, 17], [2, 16], [3, 15], [4, 14], [5, 13], [6, 12], [7, 11], [8, 10],
             [18, 27], [19, 26], [20, 25], [21, 24], [22, 23],
             [32, 36], [33, 35],
             [37, 46], [38, 45], [39, 44], [40, 43], [41, 48], [42, 47],
             [49, 55], [50, 54], [51, 53], [62, 64], [61, 65], [68, 66], [59, 57], [60, 56]),
    "AFLW": ([1, 6], [2, 5], [3, 4],
             [7, 12], [8, 11], [9, 10],
             [13, 15],
             [16, 18]),
    "COFW": ([1, 2], [5, 7], [3, 4], [6, 8], [9, 10], [11, 12], [13, 15], [17, 18], [14, 16], [19, 20], [23, 24]),
    "WFLW": ([0, 32], [1, 31], [2, 30], [3, 29], [4, 28], [5, 27], [6, 26], [7, 25], [8, 24], [9, 23], [10, 22],
             [11, 21], [12, 20], [13, 19], [14, 18], [15, 17],  # check
             [33, 46], [34, 45], [35, 44], [36, 43], [37, 42], [38, 50], [39, 49], [40, 48], [41, 47],  # elbrow
             [60, 72], [61, 71], [62, 70], [63, 69], [64, 68], [65, 75], [66, 74], [67, 73],
             [55, 59], [56, 58],
             [76, 82], [77, 81], [78, 80], [87, 83], [86, 84],
             [88, 92], [89, 91], [95, 93], [96, 97])}

num_classes = {"300W": 68, "AFLW": 19, "COFW": 29, "WFLW": 98}


class HorizontalFlip:
    def __init__(self):
        pass

    @staticmethod
    def get_transformation_matrix(img_size):
        iw, ih = img_size
        T = np.array([[-1, 0, iw],
                      [0, 1, 0],
                      [0, 0, 1]])
        return T


class SequentialTransform:
    def __init__(self, geometric_transforms, geometric_transform_prob,
                 color_distortions, color_distortions_prob, out_size,
                 shuffle=True, color_mode='bgr', interpolation=cv2.INTER_AREA,
                 border_mode=cv2.BORDER_CONSTANT, border_value=(114, 114, 114),
                 flip_point_pairs="WFLW"):
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

        if flip_point_pairs=="WFLW":
            self.flip_matrix = np.identity(num_classes["WFLW"])
            for i, j in MATCHED_PARTS["WFLW"]:
                self.flip_matrix[i, i] = 0
                self.flip_matrix[j, j] = 0
                self.flip_matrix[i, j] = 1
                self.flip_matrix[j, i] = 1

        elif flip_point_pairs in ("300W", "AFLW", "COFW"):
            self.flip_matrix = np.identity(num_classes[flip_point_pairs])
            for i, j in MATCHED_PARTS[flip_point_pairs]:
                self.flip_matrix[i-1, i-1] = 0
                self.flip_matrix[j-1, j-1] = 0
                self.flip_matrix[i-1, j-1] = 1
                self.flip_matrix[j-1, i-1] = 1
        self.flip_point = False

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
                if isinstance(transform, HorizontalFlip):
                    self.flip_point = True
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

        if self.flip_point:
            points = np.matmul(self.flip_matrix, points)
            self.flip_point = False

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
        center = iw / 2, ih / 2
        angle = random.uniform(*self.angle_range)
        scale = random.uniform(*self.scale_range)

        T = cv2.getRotationMatrix2D(center, angle, scale)
        T = np.vstack((T, np.array([[0, 0, 1]])))
        return T
