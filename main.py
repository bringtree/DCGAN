""" the train code """
import tensorflow as tf
import time
import numpy as np
import re
import os
import argparse
from model import generator, discriminator
from ops_image import *
from ops_tf_recorder import *


def main(_):
    # 获取数据集文件路径以及模型文件的路径
    train_file_path = os.path.join(FLAGS.data_dir, "train3.tfrecords")
    ckpt_path = os.path.join(FLAGS.model_dir, "model.ckpt")

    # 定义模型的2个输入槽
    model_random_number = tf.placeholder(tf.float32, [None, 100])
    model_real_image_input = tf.placeholder(tf.float32, [64, 64, 64, 3])

    # 引入2个模型
    # 生成假图片
    fake_image = generator(model_random_number)
    # 判别器对真图片的判断
    d_model_real = discriminator(model_real_image_input)
    # 判别器对假图片的判断
    d_model_fake = discriminator(fake_image, True)

    # 定义生成器模型的损失函数以及对抗模型的损失函数

    # tf.ones_like(d_model_real) * 0.9)*log(sigmoid(d_model_real)
    # + (1-tf.ones_like(d_model_real) * 0.9)*log(1-sigmoid(d_model_real))
    real_discriminator = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_model_real, labels=tf.ones_like(d_model_real) * 0.9))

    # tf.zeros_like(d_model_fake)*log(sigmoid(d_model_fake))
    # + (1-tf.zeros_like(d_model_fake)*log(1-sigmoid(d_model_fake))
    fake_discriminator = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_model_fake, labels=tf.zeros_like(d_model_fake)))

    d_loss = real_discriminator + fake_discriminator
    g_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_model_fake, labels=tf.ones_like(d_model_fake))
    )

    # 将要训练的参数进行分类,分成discriminator_var以及generator_var
    all_vars = tf.trainable_variables()
    discriminator_var = []
    generator_var = []
    for var in all_vars:
        if 'generator' in var.name:
            generator_var.append(var)

        if 'discriminator' in var.name:
            discriminator_var.append(var)

    # 获得Saver类
    saver = tf.train.Saver()

    # 训练环境的配置
    config = tf.ConfigProto(
        device_count={'gpu': 4}
    )

    with tf.Session(config=config) as sess:
        # 定义优化器
        discriminator_optimizer = tf.train.AdamOptimizer(0.0001, beta1=0.5).minimize(d_loss, var_list=discriminator_var)
        generator_optimizer = tf.train.AdamOptimizer(0.0001, beta1=0.5).minimize(g_loss, var_list=generator_var)

        # 初始化变量
        sess.run(tf.global_variables_initializer())

        # 设定batch_size大小 batch_size = 图片数量/64
        batch_size = 800

        # 读取训练数据，以及定义img_batch队列
        data = read_and_decode(train_file_path)
        coord = tf.train.Coordinator()
        img_batch = tf.train.shuffle_batch([data],
                                           batch_size=64, capacity=3000,
                                           min_after_dequeue=1000)
        # 启动队列
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # 保存状态到./logs 方便启动tensorboard
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter('./logs',
                                             sess.graph)

        # 获取保存好的模型的信息
        ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            start_epoch = int(re.search('(?<=model.ckpt-)\d+', ckpt.model_checkpoint_path).group())
            # 载入保存好的模型
            saver.restore(sess, os.path.join(ckpt.model_checkpoint_path))
            print('loading suceess')
        else:
            # 第一次训练则epoch设置为0
            start_epoch = 0
            print('loading fail')
        for epoch in range(start_epoch + 1, 400):
            for batch_index in range(0, batch_size):
                # 从img_batch队列中获取到数据到batch_image
                batch_image = sess.run(img_batch)

                # 生成随机[64,100]数据
                random_number = np.random.uniform(-1, 1, size=(64, 100)).astype(np.float32)

                # 训练D模型
                sess.run(discriminator_optimizer,
                         feed_dict={model_real_image_input: batch_image, model_random_number: random_number})
                # 训练G模型
                sess.run(generator_optimizer,
                         feed_dict={model_random_number: random_number})
                # 再次训练G模型
                sess.run(generator_optimizer,
                         feed_dict={model_random_number: random_number})

            if (epoch % 10 == 0):
                # 获取损失函数的输出
                d = fake_discriminator.eval({model_random_number: random_number})
                d2 = real_discriminator.eval({model_real_image_input: batch_image})
                g = g_loss.eval({model_random_number: random_number})
                print('epoch:' + str(epoch) + ' d_loss:' + str(d), ' d2_loss:' + str(d2), ' g_loss:' + str(g))
                print(time.strftime("%m-%d %H:%M:%S", time.localtime()))
                # 保存模型
                saver.save(sess, ckpt_path, global_step=epoch)
        # 关闭队列
        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    # 解析训练的时候附带的参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./input',
                        help='input output path')
    parser.add_argument('--model_dir', type=str, default='./output',
                        help='output model path')
    FLAGS, _ = parser.parse_known_args()
    tf.app.run(main=main)
