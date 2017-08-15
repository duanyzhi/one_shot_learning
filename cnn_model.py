import tensorflow as tf
import numpy as np

""""
  CNN Model:
    input:28*28    [batch_size, 28, 28, 1]        output: [batch_size, 1,1, 64]
    layer1-4: weight:[3,3,1,64]   strides=[1,1,1,1] # 卷积核大小[3, 3]  输入1维  输出64维
            normalization
            ReLU
            max-pooling:[2,2]  strides=[1,2,2,1]
    layer1:  output:[batch_size, 14,14, 64]
    layer2:  output:[batch_size, 7,7, 64]
    layer3:  output:[batch_size, 3,3, 64]
    layer4:  output:[batch_size, 1,1, 64]

"""


# 定义卷积，卷积后尺寸不变
def conv(input, weight, biases, offset, scale, strides=1):
    conv_conv = tf.nn.conv2d(input, weight, strides=[1, strides, strides, 1], padding='SAME') + biases
    mean, variance = tf.nn.moments(conv_conv, [0, 1, 2])
    conv_batch = tf.nn.batch_normalization(conv_conv, mean, variance, offset, scale, 1e-10)
    return tf.nn.relu(conv_batch)


# 池化，大小k*k
def max_pool(x, ksize=(2, 2), stride=(2, 2)):
    return tf.nn.max_pool(x, ksize=[1, ksize[0], ksize[1], 1],
                          strides=[1, stride[0], stride[1], 1], padding='VALID')


"""
  通常共享变量的方法就是在单独的代码块中来创建同一个模型的参数并且通过使用他们的函数
  tf.get_variable(<name>, <shape>, <initializer>): 通过所给的名字创建或是返回一个变量.
          tf.get_variable(name, shape, dtype, initializer)
  tf.variable_scope(<scope_name>): 通过 tf.get_variable()为变量名指定命名空间
  tf.get_variable()会检测已经存在的变量是否已经共享.如果你想共享他们，通过reuse_variables()这个方法来指定
"""


class cnn:  # cnn模块要复用很多次,因此要共享变量
    def __init__(self):
        self.reuse = False
        # print("init")

    def __call__(self, input, scope, reuse=False):  # __call__ :再次执行程序时调用，很好用  scope  定义名字
        self.reuse = reuse
        with tf.variable_scope(scope, reuse=self.reuse) as scope_name:
            if self.reuse: scope_name.reuse_variables()  # 共享变量,用reuse_variables()这个方法来指定
            with tf.variable_scope('conv'):
                with tf.variable_scope('conv1'):
                    self.weight_1 = tf.get_variable('weight_1', [3, 3, 1, 64])
                    self.biases_1 = tf.get_variable('biases_1', [64])
                    self.offset_1 = tf.get_variable('offset_1', [64], initializer=tf.constant_initializer(0.0))
                    self.scale_1 = tf.get_variable('scale_1', [64], initializer=tf.constant_initializer(1.0))
                    conv1_relu_1 = conv(input, self.weight_1, self.biases_1, self.offset_1, self.scale_1, strides=1)
                    conv1 = max_pool(conv1_relu_1, ksize=(2, 2), stride=(2, 2))
                    # print("conv1", conv1.shape)
                with tf.variable_scope('conv2'):
                    self.weight_2 = tf.get_variable('weight_2', [3, 3, 64, 64])
                    self.biases_2 = tf.get_variable('biases_2', [64])
                    self.offset_2 = tf.get_variable('offset_2', [64], initializer=tf.constant_initializer(0.0))
                    self.scale_2 = tf.get_variable('scale_2', [64], initializer=tf.constant_initializer(1.0))
                    conv2_relu = conv(conv1, self.weight_2, self.biases_2,  self.offset_2, self.scale_2, strides=1)
                    conv2 = max_pool(conv2_relu, ksize=(2, 2), stride=(2, 2))
                    # print("conv2", conv2.shape)
                with tf.variable_scope('conv3'):
                    self.weight_3 = tf.get_variable('weight_3', [3, 3, 64, 64])
                    self.biases_3 = tf.get_variable('biases_3', [64])
                    self.offset_3 = tf.get_variable('offset_3', [64], initializer=tf.constant_initializer(0.0))
                    self.scale_3 = tf.get_variable('scale_3', [64], initializer=tf.constant_initializer(1.0))
                    conv3_relu = conv(conv2, self.weight_3, self.biases_3,  self.offset_3, self.scale_3, strides=1)
                    conv3 = max_pool(conv3_relu, ksize=(2, 2), stride=(2, 2))
                    # print("conv3", conv3.shape)
                with tf.variable_scope('conv4'):
                    self.weight_4 = tf.get_variable('weight_4', [3, 3, 64, 64])
                    self.biases_4 = tf.get_variable('biases_4', [64])
                    self.offset_4 = tf.get_variable('offset_4', [64], initializer=tf.constant_initializer(0.0))
                    self.scale_4 = tf.get_variable('scale_4', [64], initializer=tf.constant_initializer(1.0))
                    conv4_relu = conv(conv3, self.weight_4, self.biases_4, self.offset_4, self.scale_4, strides=1)
                    conv4 = max_pool(conv4_relu, ksize=(2, 2), stride=(2, 2))
                    # print("conv4", conv4.shape)
            out = tf.squeeze(conv4, [1, 2])
        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='cnn')
        # tf.Optimizer只优化tf.GraphKeys.TRAINABLE_VARIABLES中的变量
        return out
        # class cnn():                                                  # cnn模块要复用很多次,因此要共享变量
        #     def __init__(self):
        #         print("init")
        #         self.weights = {
        #             'wc1': tf.Variable(tf.random_normal([3, 3, 1, 64])),
        #             'wc2': tf.Variable(tf.random_normal([3, 3, 64, 64])),
        #             'wc3': tf.Variable(tf.random_normal([3, 3, 64, 64])),
        #             'wc4': tf.Variable(tf.random_normal([3, 3, 64, 64])),
        #         }
        #         self.biasese = {
        #             'bc1': tf.Variable(tf.random_normal([64])),
        #             'bc2': tf.Variable(tf.random_normal([64])),
        #             'bc3': tf.Variable(tf.random_normal([64])),
        #             'bc4': tf.Variable(tf.random_normal([64])),
        #         }
        #
        #     def run_cnn(self, input):
        #         self.input = input
        #
        #         conv1 = conv(self.input, self.weights['wc1'], self.biasese['bc1'])
        #         conv1 = max_pool(conv1, ksize=(2, 2), stride=(2, 2))
        #         print("conv1",conv1.shape)
        #         conv2 = conv(conv1, self.weights['wc2'], self.biasese['bc2'])
        #         conv2 = max_pool(conv2, ksize=(2, 2), stride=(2, 2))
        #         print("conv2", conv2.shape)
        #
        #         conv3 = conv(conv2, self.weights['wc3'], self.biasese['bc3'])
        #         conv3 = max_pool(conv3, ksize=(2, 2), stride=(2, 2))
        #         print("conv3", conv3.shape)
        #
        #         conv4 = conv(conv3, self.weights['wc4'], self.biasese['bc4'])
        #         out = max_pool(conv4, ksize=(2, 2), stride=(2, 2))
        #         print("out", out.shape)
        #         return out
