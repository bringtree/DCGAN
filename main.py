import tensorflow as tf
import DCGAN_model as DCGAN

random_number = tf.placeholder(tf.float32, [64, 100])
real_image_input = tf.placeholder(tf.float32, [64, 64, 64, 3])

fake_image = DCGAN.generator_model(random_number)
real_sigmoid = DCGAN.discriminator(real_image_input)
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
    discriminator_var.append(var)
  if 'discriminator' in var.name:
    generator_var.append(var)

