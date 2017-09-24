from glob import glob
import os
import numpy as np

from scipy import misc

from PIL import Image


def load_images_src(src):
  """
  get all pictures' src,must be jpg
  :param src: the pictures' resource
  :return: return a List which is filled of all pictures' src
  """

  data = glob(os.path.join(src, '*.jpg'))
  return data


def get_image(src):
  """
  read the image, make image from [-255,255] to [-1,1]
  :param src:the picture resource
  :return: return a image, shape[64,64,3] between [-1,1]
  """

  return misc.imread(src).astype(np.float32) / 127.5 - 1


def resize(input_src, output_src):
  """
  resize the pictures which is from input_src and put them to the output_src
  :param input_src: the pictures' src ,like '/Users/huangpeisong/Desktop/project/data/new_P/*jpg'
  :param output_src: the pictures' src ,like '/Users/huangpeisong/Desktop/project/data/new_P/'
  :return:
  """
  for file in glob(input_src):
    input_file_path, input_file_all_name = os.path.split(file)
    input_file_name_pre = os.path.splitext(input_file_all_name)

    if (os.path.isdir(output_src) == False):
      os.mkdir(output_src)
    Image.open(file).resize((int(64), int(64))).save(output_src + input_file_name_pre + '.jpg')


def save_images(src, data):
  """

  :param src: like ./test.jpg
  :param data: shape(64,64,64,3)
  :return:
  """
  img = np.zeros((64 * 8, 64 * 8, 3))

  for idx, image in enumerate(data):
    i = idx % 8
    j = idx // 8
    img[j * 64:j * 64 + 64, i * 64:i * 64 + 64] = image

  misc.imsave(src, img)

# image = np.squeeze(merge(data, 64))
