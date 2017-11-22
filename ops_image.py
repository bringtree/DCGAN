"""Image globbing utility."""

import numpy as np
from scipy import misc
import tensorflow as tf
from glob import *
import os


def load_images_src(src):
    """
    get all pictures' src,must be jpg
    :param src: the pictures' resource
    :return: return a List filled with all the pictures of src
    """

    data = glob(os.path.join(src, '*.jpg'))
    return data


def get_image(src):
    """
    transform the image from int(0-255) to double(-1,1)
    :param src: the picture's resource
    :return: the np.float array, shape(64*64)
    """
    return misc.imread(src).astype(np.float32) / 127.5 - 1


def save_images(src, data):
    """
    transform the 8 images' data to a big image.
    :param src: the images' resource
    :param data:the 64 images array [64,64,63,3]
    :return: the big image consists of 8 images
    """
    img = np.zeros((64 * 8, 64 * 8, 3))

    for idx, image in enumerate(data):
        i = idx % 8
        j = idx // 8
        img[j * 64:j * 64 + 64, i * 64:i * 64 + 64] = image

    misc.imsave(src, img)


