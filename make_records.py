""" make the train records"""

from ops_image import *
from PIL import Image


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def main():
    writer = tf.python_io.TFRecordWriter("train3.tfrecords")
    # 加载图片路径
    data = load_images_src('./icon/')
    train_images = ([image for image in data])

    # 将数据写入tf.records
    for img_path in train_images:
        # 打开图片数据
        img = Image.open(img_path)
        img = img.resize((64, 64))
        # 将图片数据转化成bytes
        img_raw = img.tobytes()
        example = tf.train.Example(features=tf.train.Features(feature={
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
        }))
        # 写入数据
        writer.write(example.SerializeToString())

    writer.close()


if __name__ == '__main__':
    tf.app.run()
