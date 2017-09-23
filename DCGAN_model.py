import tensorflow as tf


def generator_model(self, random_number):
  """
  the generator model, it wants to generator 64 pictures (size:64*64*3).

  structure:
  deconvolute_pre_layer   64*100 -> 64*16384 --(reshape)-> 64*4*4*1024
  deconvolute_layer1      64*4*4*1024 -> 64*8*8*512
  deconvolute_layer2      64*8*8*512 -> 64*16*16*256
  deconvolute_layer3      64*16*16*256 -> 64*32*32*128
  deconvolute_layer4      64*32*32*128 -> 64*64*64*3

  :param random_number: the random number , shape = [64,100]

  :return: the number ,shape = [64,64,64,3], it is on behalf of 64 pictures(size:64*64*3).
  """

  with tf.variable_scope('generator_model'):
    with tf.variable_scope('deconvolute_pre_layer'):
      w = tf.get_variable(name='w', shape=[64, 16384], dtype=tf.float32,
                          initializer=tf.random_normal_initializer(stddev=0.02))
      b = tf.get_variable(name='b', shape=[16384], dtype=tf.float32,
                          initializer=tf.random_normal_initializer(stddev=0.02))
      deconv_pre = tf.matmul(w, random_number) + b
      deconv_pre = tf.reshape(deconv_pre, [64, 4, 4, *1024])
      deconv_pre = tf.nn.selu(deconv_pre)

    with tf.variable_scope('deconvolute_layer1'):
      w1 = tf.get_variable(name='w1', shape=[5, 5, 512, 1024], dtype=tf.float32,
                           initializer=tf.random_normal_initializer(stddev=0.02))
      b1 = tf.get_variable(name='b1', shape=[512], dtype=tf.float32,
                           initializer=tf.random_normal_initializer(stddev=0.02))
      deconv1 = tf.nn.conv2d_transpose(deconv_pre, w1, output_shape=[64, 8, 8, 512], strides=[1, 2, 2, 1])
      deconv1 = deconv1 + b1
      deconv1 = tf.nn.selu(deconv1)

    with tf.variable_scope('deconvolute_layer2'):
      w2 = tf.get_variable(name='w2', shape=[5, 5, 256, 512], dtype=tf.float32,
                           initializer=tf.random_normal_initializer(stddev=0.02))
      b2 = tf.get_variable(name='b2', shape=[256], dtype=tf.float32,
                           initializer=tf.random_normal_initializer(stddev=0.02))
      deconv2 = tf.nn.conv2d_transpose(deconv1, w2, output_shape=[64, 16, 16, 256], strides=[1, 2, 2, 1])
      deconv2 = deconv2 + b2
      deconv2 = tf.nn.selu(deconv2)

    with tf.variable_scope('deconvolute_layer3'):
      w3 = tf.get_variable(name='w3', shape=[5, 5, 128, 256], dtype=tf.float32,
                           initializer=tf.random_normal_initializer(stddev=0.02))
      b3 = tf.get_variable(name='b3', shape=[128], dtype=tf.float32,
                           initializer=tf.random_normal_initializer(stddev=0.02))
      deconv3 = tf.nn.conv2d_transpose(deconv2, w3, output_shape=[64, 32, 32, 128], strides=[1, 2, 2, 1])
      deconv3 = deconv3 + b3
      deconv3 = tf.nn.selu(deconv3)

    with tf.variable_scope('deconvolute_layer4'):
      w4 = tf.get_variable(name='w4', shape=[5, 5, 3, 128], dtype=tf.float32,
                           initializer=tf.random_normal_initializer(stddev=0.02))
      b4 = tf.get_variable(name='b4', shape=[3], dtype=tf.float32,
                           initializer=tf.random_normal_initializer(stddev=0.02))
      deconv4 = tf.nn.conv2d_transpose(deconv3, w4, output_shape=[64, 64, 64, 3], strides=[1, 2, 2, 1])
      deconv4 = deconv4 + b4
      deconv4 = tf.nn.tanh(deconv4)

    return deconv4


def discriminator(self, image):
  """

  structure:
  convolute_layer1 64*64*64*3  -> 64*32*32*64
  convolute_layer2 64*32*32*64 -> 64*16*16*128
  convolute_layer3 64*16*16*128 -> 64*8*8*256
  convolute_layer4 64*8*8*256 -> 64*4*4*512
  convolute_layer5 64*4*4*512 -> 64*8192 --(reshape)-> 64*64

  :param image: 64 pictures (size:64*64*3)

  :return:the number ,shape [64,64]
  """

  with tf.variable_scope('discriminator'):
    with tf.variable_scope('convolute_layer1'):
      w1 = tf.get_variable('w1', shape=[5, 5, 3, 64], dtype=tf.float32)
      b1 = tf.get_variable('b1', shape=[64], dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=0.0))

      conv_1 = tf.nn.conv2d(input=image, filter=w1, strides=[1, 2, 2, 1], padding="SAME")
      conv_1 = conv_1 + b1
      conv_1 = tf.contrib.layers.batch_norm(conv_1, decay=0.9, epsilon=1e-5, updates_collections=None,
                                            scale=True, is_training=True)
      conv_1 = tf.maximum(conv_1, conv_1 * 0.2)

    with tf.variable_scope('convolute_layer2'):
      w2 = tf.get_variable('w2', shape=[5, 5, 64, 128], dtype=tf.float32)
      b2 = tf.get_variable('b2', shape=[128], dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=0.0))

      conv_2 = tf.nn.conv2d(input=conv_1, filter=w2, strides=[1, 2, 2, 1], padding="SAME")
      conv_2 = conv_2 + b2
      conv_2 = tf.contrib.layers.batch_norm(conv_2, decay=0.9, epsilon=1e-5, updates_collections=None,
                                            scale=True, is_training=True)
      conv_2 = tf.maximum(conv_2, conv_2 * 0.2)

    with tf.variable_scope('convolute_layer3'):
      w3 = tf.get_variable('w3', shape=[5, 5, 128, 256], dtype=tf.float32)
      b3 = tf.get_variable('b3', shape=[256], dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=0.0))
      conv_3 = tf.nn.conv2d(input=conv_2, filter=w3, strides=[1, 2, 2, 1], padding="SAME")
      conv_3 = conv_3 + b3
      conv_3 = tf.contrib.layers.batch_norm(conv_3, decay=0.9, epsilon=1e-5, updates_collections=None,
                                            scale=True, is_training=True)

    with tf.variable_scope('convolute_layer4'):
      w4 = tf.get_variable('w4', shape=[5, 5, 256, 512], dtype=tf.float32)
      b4 = tf.get_variable('b4', shape=[512], dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=0.0))
      conv_4 = tf.nn.conv2d(input=conv_3, filter=w4, strides=[1, 2, 2, 1], padding="SAME")
      conv_4 = conv_4 + b4
      conv_4 = tf.contrib.layers.batch_norm(conv_4, decay=0.9, epsilon=1e-5, updates_collections=None,
                                            scale=True, is_training=True)

    with tf.variable_scope('convolute_layer5'):
      conv_5 = tf.reshape(conv_4, [64, 8192])
      w5 = tf.get_variable('w5', shape=[8192, 1], dtype=tf.float32,
                           initializer=tf.random_normal_initializer(stddev=0.0))
      b5 = tf.get_variable('b5', shape=[64], dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=0.0))
      conv_5 = tf.matmul(w5, conv_5) + b5
      conv_5 = tf.sigmoid(conv_5)

      return conv_5
