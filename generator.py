import tensorflow as tf
import numpy as np
import re
import os
import argparse
from model import generator_model
from utils import save_images


def main(_):
  model_random_number = tf.placeholder(tf.float32, [None, 100])
  fake_image = generator_model(model_random_number)

  all_vars = tf.trainable_variables()
  discriminator_var = []
  generator_var = []
  for var in all_vars:
    if 'generator' in var.name:
      generator_var.append(var)
    if 'discriminator' in var.name:
      discriminator_var.append(var)

  saver = tf.train.Saver()

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
    if ckpt and ckpt.model_checkpoint_path:
      start_epoch = int(re.search('(?<=model.ckpt-)\d+', ckpt.model_checkpoint_path).group())
      saver.restore(sess, os.path.join(ckpt.model_checkpoint_path))
      print('loading suceess')

      for batch_index in range(0, 10):
        random_number = np.random.uniform(-1, 1, size=(64, 100)).astype(np.float32)
        save_image_data = sess.run(fake_image, feed_dict={model_random_number: random_number})
        save_images('./fake_image/' + 'epoch_' + str(start_epoch) + 'batch_index_' + str(batch_index) + '.jpg',
                    save_image_data)

    else:
      print('loading fail')


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='./input',
                      help='input output path')
  parser.add_argument('--model_dir', type=str, default='./output',
                      help='output model path')
  FLAGS, _ = parser.parse_known_args()
  tf.app.run(main=main)
