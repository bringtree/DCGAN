""" recorder globbing utility """
import tensorflow as tf


def read_and_decode(filename_queue):
    """
    read the data of recorders, and decode it.
    :param filename_queue:the recorders's resource
    :return:the data(type:tf.float scope:[-1,1])
    """
    filename_queue = tf.train.string_input_producer([filename_queue])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                       })

    image = tf.decode_raw(features['img_raw'], tf.uint8)
    image = tf.reshape(image, [64, 64, 3])
    image = tf.cast(image, tf.float32) * (1. / 255) - 0.5

    return image
