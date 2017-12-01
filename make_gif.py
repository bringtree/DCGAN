"""Used to generator gif photo"""

import os
import imageio

images_list = []


# 定义函数将生成的图片路径都放到一个list里面
def generator_64():
    for i in range(1, 20):
        for j in range(0, 3):
            images_list.append(
                os.getcwd() + '/fake_image/epoch_%dbatch_index_%d.jpg' % (i * 2, j * 200))
            images_list.append(
                os.getcwd() + '/fake_image/epoch_%dbatch_index_%d.jpg' % (i * 2, j * 200))


generator_64()

# 将所有图片放到一个list里面
images = []
for filename in images_list:
    images.append(imageio.imread(filename))

# 调用mimsave函数把图片存放在一个gif里面
imageio.mimsave('48*48.gif', images)
