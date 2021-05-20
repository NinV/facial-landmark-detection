import pathlib
from torch.utils.data import Dataset
from libs.utils.image import load_image, letterbox, simple_normalize
from libs.utils.image import Resize


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
        self.resize_func = Resize(image_size)
        self.image_size = image_size

        # using dictionary instead of list for cases where some image files cannot be read
        self.images = {}
        self._image_ids = []  # for holding image_id(s) of valid image_files
        self._load_images()

    def _load_images(self):
        raise NotImplemented

    def __len__(self):
        return len(self._image_ids)
