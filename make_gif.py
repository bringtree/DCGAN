"""Used to generator gif photo"""

import os
import imageio

images_list = []


def generator_48():
    for i in range(0, 40):
        for j in range(1, 3):
            images_list.append('/Users/huangpeisong/Desktop/课设/f64/samples/train_%02d_%04d.png' % (i, j * 200 - 1))
            images_list.append('/Users/huangpeisong/Desktop/课设/f64/samples/train_%02d_%04d.png' % (i, j * 200 - 1))


def generator_64():
    for i in range(1, 20):
        for j in range(0, 3):
            images_list.append(
                '/Users/huangpeisong/Desktop/课设/fake_image/epoch_%dbatch_index_%d.jpg' % (i * 2, j * 200))
            images_list.append(
                '/Users/huangpeisong/Desktop/课设/fake_image/epoch_%dbatch_index_%d.jpg' % (i * 2, j * 200))


generator_48()
images = []
for filename in images_list:
    images.append(imageio.imread(filename))

imageio.mimsave('48*48.gif', images)
