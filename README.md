数据结构课设

基于https://github.com/carpedm20/DCGAN-tensorflow 的修改

数据来源 是知乎上 的 何之源的 数据 然后 缩放到64*64 放置在 data目录下解压 下载地址https://pan.baidu.com/s/1o8wqJTg 密码eecd
#### 目录说明
```
├── fake_image(存放生成图片的代码)
├── input(数据存放地方)
│   └── train3.tfrecords
├── main.py(训练的代码)
├── make_records.py(制造数据集的代码)
├── model.py(模型的代码)
├── output(模型输出的代码)
│   ├── checkpoint
│   ├── model.ckpt-110.data-00000-of-00001
│   ├── model.ckpt-110.index
│   ├── model.ckpt-110.meta
│   ├── model.ckpt-230.data-00000-of-00001
│   ├── model.ckpt-230.index
│   ├── model.ckpt-230.meta
│   ├── model.ckpt-300.data-00000-of-00001
│   ├── model.ckpt-300.index
│   └── model.ckpt-300.meta
├── generator.py(根据训练好的模型生成图片的代码)
└── utils.py(一些用到的工具代码)
```

#### 模型调参的一些事：
###### 1使用SELU函数调参是 在前期(1,2的时候epochs)的话会表现出来的情况会比 BN+Relu强，但是到后面(大概到30epoch左右就不会多大变化了，生成图片的效果还是不佳的，而且超级容易崩，就是生成的图片和你随机生成的数字相关度不高，图片都是一模一样的)
###### 2当出现崩的情况 一个解决方法就是 把判别器判断真实图片的那个交叉商那里，不让他把真实图片当作1，把它降低一丢丢。或者加个噪点，或者偶数次就0。9，奇数次就1.0.随缘调，就是不让他恒等1。
###### 3使用relu+bn的话 差不多在60的时候就不动了，不过结果看上去比selu强

