from glob import glob
import os
import numpy

from scipy import misc


class file_operas(object):
  def load_images_src(self, src):
    """
    get all pictures' src,must be jpg
    :param src: the pictures' resource
    :return: return a List which is filled of all pictures' src
    """

    data = glob(os.path.join(src, '*jpg'))
    return data

  def get_image(self, src):
    """
    read the image, make image from [-255,255] to [-1,1]
    :param src:the picture resource
    :return: return a image, shape[64,64,3] between [-1,1]
    """

    return misc.imread(src).astype(numpy.float32) / 127.5 - 1

