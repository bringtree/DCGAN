import tensorflow as tf
import DCGAN_model as DCGAN
import file_operas as file_operas
import time
import numpy as np
import re
import os

# build model
model_random_number = tf.placeholder(tf.float32, [64, 100])
model_real_image_input = tf.placeholder(tf.float32, [64, 64, 64, 3])

fake_image = DCGAN.generator_model(model_random_number)
real_sigmoid = DCGAN.discriminator(model_real_image_input)
fake_sigmoid = DCGAN.discriminator(fake_image, True)

real_discriminator = tf.reduce_mean(
  tf.nn.sigmoid_cross_entropy_with_logits(logits=real_sigmoid, labels=tf.ones_like(real_sigmoid)))

fake_discriminator = tf.reduce_mean(
  tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_sigmoid, labels=tf.zeros_like(fake_sigmoid)))

d_loss = real_discriminator + fake_discriminator

g_loss = tf.reduce_mean(
  tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_sigmoid, labels=tf.ones_like(fake_sigmoid))
)

all_vars = tf.trainable_variables()
discriminator_var = []
generator_var = []
for var in all_vars:
  if 'generator' in var.name:
    generator_var.append(var)
  if 'discriminator' in var.name:
    discriminator_var.append(var)

discriminator_optimizer = tf.train.AdamOptimizer(0.0002).minimize(d_loss, var_list=discriminator_var)
generator_optimizer = tf.train.AdamOptimizer(0.0002).minimize(g_loss, var_list=generator_var)
saver = tf.train.Saver()
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  data = file_operas.load_images_src('./data/new_p/')
  batch_size = len(data) // 64

  ckpt = tf.train.get_checkpoint_state('./checkpoint/')
  if ckpt and ckpt.model_checkpoint_path:
    start_epoch = int(re.search('(?<=DCGAN.ckpt-)\d+', ckpt.model_checkpoint_path).group())
    saver.restore(sess, os.path.join(ckpt.model_checkpoint_path))
    print('loading suceess')
  else:
    start_epoch = 0
    print('loading fail')
  for epoch in range(start_epoch + 1, 50):

    for batch_index in range(0, batch_size):
      batch_image = np.array([
        file_operas.get_image(image) for image in data[batch_index * 64:(batch_index + 1) * 64]
      ]).astype(np.float32)

      random_number = np.random.uniform(-1, 1, size=(64, 100)).astype(np.float32)

      sess.run(discriminator_optimizer,
               feed_dict={model_real_image_input: batch_image, model_random_number: random_number})
      sess.run(generator_optimizer,
               feed_dict={model_random_number: random_number})
      sess.run(generator_optimizer,
               feed_dict={model_random_number: random_number})

      print(time.strftime("%m-%d %H:%M:%S", time.localtime()))

      if (batch_index % 100 == 0):
        d = d_loss.eval({model_real_image_input: batch_image, model_random_number: random_number})
        g = g_loss.eval({model_random_number: random_number})
        print('d_loss:' + str(d), 'g_loss:' + str(g))

        save_image_data = sess.run(fake_image, feed_dict={model_random_number: random_number})
        file_operas.save_images('./fake_image/' + 'epoch_' + str(epoch) + 'batch_index_' + str(batch_index) + '.jpg',
                                save_image_data)

    saver.save(sess, './checkpoint/DCGAN.ckpt', global_step=epoch)
