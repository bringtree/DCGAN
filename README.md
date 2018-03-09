数据结构课设.

数据读取http://blog.csdn.net/lujiandong1/article/details/53376134

基于https://github.com/carpedm20/DCGAN-tensorflow 的修改

数据来源 是知乎上 的 何之源的 数据 然后 缩放到64*64 放置在 data目录下解压 下载地址https://pan.baidu.com/s/1o8wqJTg 密码eecd
#### 目录说明

```
├── fake_image(存放生成图片)
├── input(数据存放地方)
│   └── train3.tfrecords
├── output(存储训练的模型参数的位置)
│   ├── checkpoint
│   ├── model.ckpt-300.data-00000-of-00001
│   ├── model.ckpt-300.index
│   └── model.ckpt-300.meta
├── main.py(训练的代码)
├── make_wechat_photo.py(使用训练好的模型生成微信头像的代码)
├── make_gif.py(将生成图片制作成gif图的代码)
├── make_records.py(制作数据集的代码)
├── model.py(generator模型以及discributor模型的代码)
├── ops_image.py(一些图片操作的代码)
└── ops_tf_recorder.py(数据集操作的代码)
```

一些说明
1 generator 模型使用 tanh 激活函数 保证了 输出值在\[-1,1\],因为图片在被\[0,255\]转成了\[-1,1\]
  为什么不用sigmoid 而用tanh, 因为sigmoid 在2侧的值 的导数 变化过小。
  局部生成图片变灰 或者白 是上溢或者下溢了?

2 梯度下降法的一个demo

    var m = -1000;
    var b = 1000;
    data = [
        {x: 1, y: 1},
        {x: 2, y: 2},
        {x: 3, y: 3},
        {x: 4, y: 4}
    ];

    function grad() {
        var learning_rate = 0.05;
        for (var i = 0; i < data.length; i++) {
            var x = data[i].x;
            var y = data[i].y;
            var guess = m * x + b;
            var error = y - guess;
            m = m + error * x * learning_rate;
            b = b + error * learning_rate;
        }
    }

    for (var i = 0; i < 1000; i++)
        grad()

    console.log(m, b);

acgan使用优化器是Adam，好处学习率可以自己学习调节，另外引入moment，而且免去batch_size大小的设置烦恼。

3 geneartor模型 更新2次参数 是因为 学习弱。

4 2张看到能明白卷积计算过程的图 :
    ![](https://github.com/bringtree/DCGAN/blob/master/img/2301760-73e20657157e186f.gif)
    ![](https://github.com/bringtree/DCGAN/blob/master/img/KPyqPOB.gif)

5 反卷积的图: 注意 反卷积并不是说能还原卷积前的结果，他也只是一种计算方法。
    ![](https://github.com/bringtree/DCGAN/blob/master/img/20161017150336392)

6 模型的图:
    ![](https://github.com/bringtree/DCGAN/blob/master/img/structure_cnn.png)

7 设置多个batch优势:同时并行计算，求均值。

8 batch_norm 归一化 均值0、方差为1

  1 加快了收敛速度

  2
  ```
  原因在于神经网络学习过程本质就是为了学习数据分布，一旦训练数据与测试数据的分布不同，那么网络的泛化能力也大大降低；另外一方面，一旦每批训练数据的分布各不相同(batch 梯度下降)，那么网络就要在每次迭代都去学习适应不同的分布，这样将会大大降低网络的训练速度，这也正是为什么我们需要对数据都要做一个归一化预处理的原因。
  ```

  3（有点像condition number？）
  ```
  降低耦合度？
  对于深度网络的训练是一个复杂的过程，只要网络的前面几层发生微小的改变，那么后面几层就会被累积放大下去。一旦网络某一层的输入数据的分布发生改变，那么这一层网络就需要去适应学习这个新的数据分布，所以如果训练过程中，训练数据的分布一直在发生变化，那么将会影响网络的训练速度
  ```

  4
  ```
  比如我网络中间某一层学习到特征数据本身就分布在S型激活函数的两侧，你强制把它给我归一化处理、标准差也限制在了1，把数据变换成分布于s函数的中间部分，这样就相当于我这一层网络所学习到的特征分布被你搞坏了，这可怎么办？于是文献使出了一招惊天地泣鬼神的招式：变换重构，引入了可学习参数γ、β，这就是算法关键之处：
  ```

9 start_queue_runners 问题:
  ```
  https://zhuanlan.zhihu.com/p/27238630
  ```
  就是 启动填充队列的线程，这时系统就不再“停滞”

10 交叉熵:

   描述了 预测值 与 理想值的差距 p真log(p预测)+(1-p)*log(1-p预测)的累加，也就是在正确值上的p越大 交叉熵也就越低，混乱度低，都聚合在正确值上。
   ??? classification error 不能精确描述 ，classification error会一刀切，认为错就是错，对就是对，不能够做到聚合。(也就是在判断一个输出的结果时 0.9的概率是正确的和0.6的概率是对的的时候 在 classification error 看来都是对的，没有区别)
   另外 分类问题，常用one-hot\[one-hot不能用于作决策树，虽然和这个东西无关。。。。\] + cross entropy 组合。






