from glob import glob

import os


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

    :param src:
    :return:
    """
