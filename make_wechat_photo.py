"""generator the wechat photos """
import tensorflow as tf
import numpy as np
import re
import os
import argparse
from model import generator
from ops_image import save_images


def main(_):
    # 定义模型的2个输入槽
    model_random_number = tf.placeholder(tf.float32, [64, 100])
    fake_image = generator(model_random_number)

    all_vars = tf.trainable_variables()
    # 收集训练好的参数generator_var
    generator_var = []
    for var in all_vars:
        if 'generator' in var.name:
            generator_var.append(var)

    # 获得Saver类
    saver = tf.train.Saver()

    with tf.Session() as sess:
        # 初始化参数
        sess.run(tf.global_variables_initializer())
        # 获取模型路径
        ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            # 读取模型
            start_epoch = int(re.search('(?<=model.ckpt-)\d+', ckpt.model_checkpoint_path).group())
            saver.restore(sess, os.path.join(ckpt.model_checkpoint_path))
            print('loading model suceess')

            if (not os.path.exists(FLAGS.fake_image_dir)):
                os.mkdir(FLAGS.fake_image_dir)

            # 生成图片
            for batch_index in range(0, 10):
                random_number = np.random.uniform(-1, 1, size=(64, 100)).astype(np.float32)
                save_image_data = sess.run(fake_image, feed_dict={model_random_number: random_number})
                # 保存图片
                save_images(
                    FLAGS.fake_image_dir + '/epoch_' + str(start_epoch) + 'batch_index_' + str(batch_index) + '.jpg',
                    save_image_data)

            print('generator end')
        else:
            print('loading model fail')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default='./output',
                        help='output model path')
    parser.add_argument('--fake_image_dir', type=str, default='./fake_image',
                        help='generator fake_image_dir path')
    FLAGS, _ = parser.parse_known_args()
    tf.app.run(main=main)
