import tensorflow as tf
import time
import numpy as np
import re
import os
import argparse
from model import generator_model, discriminator
from utils import *


def main(_):
  train_file_path = os.path.join(FLAGS.data_dir, "train3.tfrecords")
  ckpt_path = os.path.join(FLAGS.model_dir, "model.ckpt")

  model_random_number = tf.placeholder(tf.float32, [None, 100])
  model_real_image_input = tf.placeholder(tf.float32, [64, 64, 64, 3])

  fake_image = generator_model(model_random_number)
  d_model_real = discriminator(model_real_image_input)
  d_model_fake = discriminator(fake_image, True)

  real_discriminator = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(logits=d_model_real, labels=tf.ones_like(d_model_real) * 0.9))

  fake_discriminator = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(logits=d_model_fake, labels=tf.zeros_like(d_model_fake)))

  d_loss = real_discriminator + fake_discriminator

  g_loss = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(logits=d_model_fake, labels=tf.ones_like(d_model_fake))
  )

  all_vars = tf.trainable_variables()
  discriminator_var = []
  generator_var = []
  for var in all_vars:
    if 'generator' in var.name:
      generator_var.append(var)

    if 'discriminator' in var.name:
      discriminator_var.append(var)

  saver = tf.train.Saver()

  config = tf.ConfigProto(
    device_count={'gpu': 4}
  )

  with tf.Session(config=config) as sess:
    discriminator_optimizer = tf.train.AdamOptimizer(0.0001, beta1=0.5).minimize(d_loss, var_list=discriminator_var)
    generator_optimizer = tf.train.AdamOptimizer(0.0001, beta1=0.5).minimize(g_loss, var_list=generator_var)
    sess.run(tf.global_variables_initializer())
    batch_size = 800

    data = read_and_decode(train_file_path)
    coord = tf.train.Coordinator()
    img_batch = tf.train.shuffle_batch([data],
                                       batch_size=64, capacity=3000,
                                       min_after_dequeue=1000)
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
    if ckpt and ckpt.model_checkpoint_path:
      start_epoch = int(re.search('(?<=model.ckpt-)\d+', ckpt.model_checkpoint_path).group())
      saver.restore(sess, os.path.join(ckpt.model_checkpoint_path))
      print('loading suceess')
    else:
      start_epoch = 0
      print('loading fail')
    for epoch in range(start_epoch + 1, 400):
      for batch_index in range(0, batch_size):
        batch_image = sess.run(img_batch)

        random_number = np.random.uniform(-1, 1, size=(64, 100)).astype(np.float32)

        sess.run(discriminator_optimizer,
                 feed_dict={model_real_image_input: batch_image, model_random_number: random_number})
        sess.run(generator_optimizer,
                 feed_dict={model_random_number: random_number})
        sess.run(generator_optimizer,
                 feed_dict={model_random_number: random_number})

      if (epoch % 10 == 0):
        d = fake_discriminator.eval({model_random_number: random_number})
        d2 = real_discriminator.eval({model_real_image_input: batch_image})
        g = g_loss.eval({model_random_number: random_number})
        print('epoch:' + str(epoch) + ' d_loss:' + str(d), ' d2_loss:' + str(d2), ' g_loss:' + str(g))
        print(time.strftime("%m-%d %H:%M:%S", time.localtime()))
        saver.save(sess, ckpt_path, global_step=epoch)
    coord.request_stop()
    coord.join(threads)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='./input',
                      help='input output path')
  parser.add_argument('--model_dir', type=str, default='./output',
                      help='output model path')
  FLAGS, _ = parser.parse_known_args()
  tf.app.run(main=main)
