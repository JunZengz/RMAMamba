import os
from glob import glob
import cv2
import numpy as np
from torch.utils.data import Dataset

def load_LiTS_data(path):
    def get_data(path, name):
        images = sorted(glob(os.path.join(path, name, "images", "*.jpg")))
        labels = sorted(glob(os.path.join(path, name, "masks", "grayscale", "liver", "*.jpg")))
        # print(f"path: {name}; images: {len(images)}; labels: {len(labels)}")

        return images, labels

    """ Names """
    dirs = sorted(os.listdir(path))
    test_names = [f"liver_{i}" for i in range(0, 30, 1)]
    valid_names = [f"liver_{i}" for i in range(30, 60, 1)]

    train_names = [item for item in dirs if item not in test_names]
    train_names = [item for item in train_names if item not in valid_names]

    """ Training data """
    train_x, train_y = [], []
    for name in train_names:
        x, y = get_data(path, name)
        train_x += x
        train_y += y

    """ Validation data """
    valid_x, valid_y = [], []
    for name in valid_names:
        x, y = get_data(path, name)
        valid_x += x
        valid_y += y

    """ Testing data """
    test_x, test_y = [], []
    for name in test_names:
        x, y = get_data(path, name)
        test_x += x
        test_y += y

    return [(train_x, train_y), (valid_x, valid_y), (test_x, test_y)]


def load_CirrMRI_data(path):
    def get_data(path, name):
        images = sorted(glob(os.path.join(path, name, "images", "*.png")))
        labels = sorted(glob(os.path.join(path, name, "masks", "*.png")))
        return images, labels

    """ Names """
    train_path = f"{path}/train"
    valid_path = f"{path}/valid"
    test_path = f"{path}/test"

    train_dirs = sorted(os.listdir(train_path))
    valid_dirs = sorted(os.listdir(valid_path))
    test_dirs = sorted(os.listdir(test_path))

    train_names = [item for item in train_dirs]
    valid_names = [item for item in valid_dirs]
    test_names = [item for item in test_dirs]

    # dirs = sorted(os.listdir(path))
    # test_names = [f"liver_{i}" for i in range(0, 30, 1)]
    # valid_names = [f"liver_{i}" for i in range(30, 60, 1)]
    # train_names = [item for item in dirs if item not in test_names]
    # train_names = [item for item in train_names if item not in valid_names]

    """ Training data """
    train_x, train_y = [], []
    for name in train_names:
        x, y = get_data(train_path, name)
        train_x += x
        train_y += y

    """ Validation data """
    valid_x, valid_y = [], []
    for name in valid_names:
        x, y = get_data(valid_path, name)
        valid_x += x
        valid_y += y

    """ Testing data """
    test_x, test_y = [], []
    for name in test_names:
        x, y = get_data(test_path, name)
        test_x += x
        test_y += y

    return [(train_x, train_y), (valid_x, valid_y), (test_x, test_y)]


class DATASET(Dataset):
    def __init__(self, images_path, masks_path, size, transform=None):
        super().__init__()

        self.images_path = images_path
        self.masks_path = masks_path
        self.size = size
        self.transform = transform
        self.n_samples = len(images_path)

    def __getitem__(self, index):
        """ Image """
        image = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR)
        mask = cv2.imread(self.masks_path[index], cv2.IMREAD_GRAYSCALE)

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        image = cv2.resize(image, self.size)
        image = np.transpose(image, (2, 0, 1))
        image = image/255.0

        mask = cv2.resize(mask, self.size)
        mask = np.expand_dims(mask, axis=0)
        mask = mask/255.0

        return image, mask

    def __len__(self):
        return self.n_samples


def get_load_data_func(dataset_name):
    assert dataset_name in ['LiTS', 'CirrMRI600+']
    if dataset_name == 'LiTS':
        return load_LiTS_data
    elif dataset_name == 'CirrMRI600+':
        return load_CirrMRI_data