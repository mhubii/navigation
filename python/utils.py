import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import os

# INPUT_SHAPE as input for CNN (cropped shapes).
IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 120, 160, 3
INPUT_SHAPE = (IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH)


class DataSetGenerator(Dataset):
    def __init__(self, data_dir, transform=None):
        """
            Load all image paths and velocities
            from the .txt file.
        """
        self.data_dir = data_dir
        self.image_paths, self.velocities = load_data(data_dir)
        self.transform = transform

    def __getitem__(self, index):
        """
            Get an image and the velocity by index.
        """
        img_left = load_image(self.image_paths[index][0])
        img_right = load_image(self.image_paths[index][1])

        vel = self.velocities[index]

        sample = {'img_left': img_left, 'img_right': img_right, 'vel': vel}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        """
            Return the length of the whole data set.
        """
        return self.velocities.shape[0]


class PreProcessData(object):
    """
        Pre-process the data.
    """
    def __call__(self, sample):
        img_left, img_right, vel = sample['img_left'], sample['img_right'], sample['vel']

        img_left = normalize(img_left)
        img_right = normalize(img_right)

        return {'img_left': img_left, 'img_right': img_right, 'vel': vel}


class ToTensor(object):
    """
        Convert data to tensor.
    """
    def __call__(self, sample):
        img_left, img_right, vel = sample['img_left'], sample['img_right'], sample['vel']

        # Change HxWxC to CxHxW.
        img_left = np.transpose(img_left, (2, 0, 1))
        img_right = np.transpose(img_right, (2, 0, 1))

        return {'img_left': torch.from_numpy(img_left).float(),
                'img_right': torch.from_numpy(img_right).float(),
                'vel': torch.from_numpy(vel).float()}


def load_data(data_dir):
    """
        Loads the input data and separates it into image_paths
        and velocities.
    :return:
        image_paths: np.ndarray
                     Location of recorded images.
        labels: float
                Velocities.
    """
    data_df = pd.read_csv(os.path.join(os.getcwd(),
                          data_dir, 'log.txt'),
                          delimiter=', ',
                          names=['left', 'right', 'vel0', 'vel1', 'vel2'],
                          engine='python')

    image_paths = data_df[['left', 'right']].values
    velocities = data_df[['vel0', 'vel1', 'vel2']].values

    return image_paths, velocities


def load_image(image_file):
    """
        Load RGB image from a file.
    """

    img_data = open(image_file, 'rb').read()
    img_size = (IMAGE_HEIGHT, IMAGE_WIDTH)

    image = np.array(Image.frombytes('RGB', img_size, img_data, 'raw'))

    return image


def normalize(image):
    """
        Normalize image to [-1, 1].
    """
    image = image/127.5 - 1.

    return image

