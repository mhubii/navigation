import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import skimage.exposure
import skimage.transform
import skimage.color
import skimage.io
import cv2
import os

# INPUT_SHAPE as input for CNN (cropped shapes).
IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 75, 320, 3
INPUT_SHAPE = (IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH)


class DataSetGenerator(Dataset):
    def __init__(self, data_dir, transform=None):
        """
            Load all image paths and steering angles
            from the .csv file.
        """
        self.data_dir = data_dir
        self.image_paths, self.steering_angles = load_data(data_dir)
        self.transform = transform

    def __getitem__(self, index):
        """
            Get an image and the steering angle by index.
            Outputs the adjusted steering angle.
        """
        img, steering_angle = select_image(self.data_dir,
                                           self.image_paths[index],
                                           self.steering_angles[index])

        sample = {'img': img, 'steering_angle': steering_angle}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        """
            Return the length of the whole data set.
        """
        return self.steering_angles.shape[0]


class PreProcessData(object):
    """
        Pre-process the data.
    """
    def __call__(self, sample):
        img, steering_angle = sample['img'], sample['steering_angle']

        img = crop(img)
        img = rgb_to_yuv(img)
        img = random_shadow(img)
        img = normalize(img)

        return {'img': img, 'steering_angle': steering_angle}


class AugmentData(object):
    """
        Augment data.
    """
    def __call__(self, sample):
        img, steering_angle = sample['img'], sample['steering_angle']

        img = random_brightness(img)
        img, steering_angle = random_flip(img, steering_angle)
        img, steering_angle = random_translate(img, steering_angle, 100, 10)

        return {'img': img, 'steering_angle': steering_angle}


class ToTensor(object):
    """
        Convert data to tensor.
    """
    def __call__(self, sample):
        img, steering_angle = sample['img'], sample['steering_angle']

        # Change HxWxC to CxHxW.
        img = np.transpose(img, (2, 0, 1))

        return {'img': torch.from_numpy(img).float(),
                'steering_angle': torch.FloatTensor([steering_angle])}


def load_data(data_dir):
    """
        Loads the input data and separates it into image_paths
        and steering_angles.penis
    :return:
        image_paths: np.ndarray
                     Location of recorded images.
        labels: float/rad
                Steering angle.
    """
    data_df = pd.read_csv(os.path.join(os.getcwd(),
                          data_dir, 'driving_log.csv'),
                          names=['center', 'left', 'right',
                                 'steering', 'throttle',
                                 'break', 'velocity'])

    image_paths = data_df[['center', 'left', 'right']].values
    steering_angles = data_df['steering'].values

    steering_angles = np.stack((steering_angles, steering_angles, steering_angles), axis=1)

    steering_angles = np.reshape(steering_angles, steering_angles.size)
    image_paths = np.reshape(image_paths, image_paths.size)

    return image_paths, steering_angles


def select_image(data_dir, image_file, steering_angle):
    """
        Load an image and adjust the steering angle accordingly
        to the location of the camera (center, left or right).
    :return:
        image: np.array
               RGB values of the image.
        steering_angle: float/rad
                        Steering angle corresponding to image.
    """
    img = load_image(data_dir, image_file)

    if 'left' in image_file:
        steering_angle += 0.2
    elif 'right' in image_file:
        steering_angle -= 0.2

    return img, steering_angle


def load_image(data_dir, image_file):
    """
        Load RGB image from a file.
    """
    return skimage.io.imread(os.path.join(os.getcwd(), data_dir, image_file))


def crop(image):
    """
        Crop of the sky since it does not add
        useful information for the training.
    """
    image = image[60:135, :]
    return image


def rgb_to_yuv(image):
    """
        Converts an image from rgb to yuv.
    """
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    return image


def normalize(image):
    """
        Normalize image to [-1, 1].
    """
    image = image/127.5 - 1.
    return image


def random_brightness(image):
    """
        Adjust brightness of image randomly by applying
        a gamma correction
            image = image^(1/gamma)
        randomly.
    """

    gamma = np.random.random_sample() + 0.5
    return skimage.exposure.adjust_gamma(image, gamma)


def random_flip(image, steering_angle):
    """
        Flips an image as there mainly left turns
        on the training ground.
    :return:
        image: array
               Flipped image.
        steering_angle: rad
                        Sign reverted steering angle.
    """
    choice = np.random.choice(2)
    if choice == 0:
        return cv2.flip(image, 1), -steering_angle
    else:
        return image, steering_angle


def random_translate(image, steering_angle, range_x, range_y):
    """
        Randomly translates an input image
    """
    trans_x = range_x * (np.random.rand() - 0.5)
    trans_y = range_y * (np.random.rand() - 0.5)
    steering_angle += trans_x * 0.002
    trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
    height, width = image.shape[:2]
    image = cv2.warpAffine(image, trans_m, (width, height))

    return image, steering_angle


def random_shadow(image):
    """
        Add random shadow as trapeze between edge and x1/x2
    """

    if random.random() <= 1:

        x1 = random.randint(50, 200)
        x2 = random.randint(x1, x1 + 80)
        while x2 == x1:
            x2 = random.randint(x1, x1 + 80)
        step = np.float32(abs(x1 - x2)) / 75.

        array = np.arange(x1, x2, step)

        if random.random() < 0.5:
            array = np.flipud(array)
        array = np.array(array, dtype=int)

        if random.random() <= 0.5:
            for i in range(IMAGE_HEIGHT):
                image[i, 0:array[i], 0] = np.random.randint(0, 5, array[i])
        else:
            for i in range(IMAGE_HEIGHT):
                image[i, array[i]:IMAGE_WIDTH, 0] = np.random.randint(0,5, IMAGE_WIDTH - array[i])

return image
