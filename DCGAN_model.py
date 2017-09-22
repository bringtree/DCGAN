import tensorflow as tf


class DCGAN_model(object):
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
                            initializer=tf.random_normal_initializer(stddev=0.2))
        b = tf.get_variable(name='b', shape=[16384], dtype=tf.float32,
                            initializer=tf.random_normal_initializer(stddev=0.2))
        deconv_pre = tf.matmul(w, random_number) + b
        deconv_pre = tf.reshape(deconv_pre, [64, 4, 4, *1024])
        deconv_pre = tf.nn.selu(deconv_pre)

      with tf.variable_scope('deconvolute_layer1'):
        w1 = tf.get_variable(name='w1', shape=[5, 5, 512, 1024], dtype=tf.float32,
                             initializer=tf.random_normal_initializer(stddev=0.2))
        b1 = tf.get_variable(name='b1', shape=[512], dtype=tf.float32,
                             initializer=tf.random_normal_initializer(stddev=0.2))
        deconv1 = tf.nn.conv2d_transpose(deconv_pre, w1, output_shape=[64, 8, 8, 512], strides=[1, 2, 2, 1])
        deconv1 = deconv1 + b1
        deconv1 = tf.nn.selu(deconv1)

      with tf.variable_scope('deconvolute_layer2'):
        w2 = tf.get_variable(name='w2', shape=[5, 5, 256, 512], dtype=tf.float32,
                             initializer=tf.random_normal_initializer(stddev=0.2))
        b2 = tf.get_variable(name='b2', shape=[256], dtype=tf.float32,
                             initializer=tf.random_normal_initializer(stddev=0.2))
        deconv2 = tf.nn.conv2d_transpose(deconv1, w2, output_shape=[64, 16, 16, 256], strides=[1, 2, 2, 1])
        deconv2 = deconv2 + b2
        deconv2 = tf.nn.selu(deconv2)

      with tf.variable_scope('deconvolute_layer3'):
        w3 = tf.get_variable(name='w3', shape=[5, 5, 128, 256], dtype=tf.float32,
                             initializer=tf.random_normal_initializer(stddev=0.2))
        b3 = tf.get_variable(name='b3', shape=[128], dtype=tf.float32,
                             initializer=tf.random_normal_initializer(stddev=0.2))
        deconv3 = tf.nn.conv2d_transpose(deconv2, w3, output_shape=[64, 32, 32, 128], strides=[1, 2, 2, 1])
        deconv3 = deconv3 + b3
        deconv3 = tf.nn.selu(deconv3)

      with tf.variable_scope('deconvolute_layer4'):
        w4 = tf.get_variable(name='w4', shape=[5, 5, 3, 128], dtype=tf.float32,
                             initializer=tf.random_normal_initializer(stddev=0.2))
        b4 = tf.get_variable(name='b4', shape=[3], dtype=tf.float32,
                             initializer=tf.random_normal_initializer(stddev=0.2))
        deconv4 = tf.nn.conv2d_transpose(deconv3, w4, output_shape=[64, 64, 64, 3], strides=[1, 2, 2, 1])
        deconv4 = deconv4 + b4
        deconv4 = tf.nn.tanh(deconv4)

      return deconv4
