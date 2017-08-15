# ------------------------------------------------------------------------
# one shot learning with Matching Networks
# Reference:Matching Networks for One Shot Learning -- Google DeepMind
# python 3.5
# tensorflow 1.1
# numpy
# 11/6/2017
# UESTC--Duan
# ------------------------------------------------------------------------
from __future__ import division
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import cnn_model
import cv2
import tqdm

from lstm_model import BidirectionalLSTM, read_lstm
from cosine_distance import cd, compute_distance
from load_data import load_data_minibatch

input_image = [28, 28]   # 28*28
ways = 5            # 5-ways  5 classes
shot = 5            # one-shot  sample 1 image for every input memory
batch_size = 32
memory_size = ways*shot   # 一个batch中memory有ways*shot个图片
channel = 1              # RGB只取一个通道
learning_rate = 0.001
training_iters = 20000
display_step = 1
if_lstm = False        # 去掉lstm效果更好
feature_size = 64

data = np.load('npy\\imagn_one.npy')
mean = np.mean(data)
std = np.std(data)
data = (data - mean)/std
data = np.reshape(data, [-1, 1300, 28, 28])     # 1600 classes has 20 examples
data = np.random.permutation(data)            # 打乱 return a permuted range
train_data = data[:1200, :, :, :]             # 1200 20 28 28
test_data = data[1200:, :, :, :]

get_minibatch_train = load_data_minibatch(train_data, batch_size, memory_size, input_image, ways, shot, reuse=False)
get_minibatch_test = load_data_minibatch(test_data, batch_size, memory_size, input_image, ways, shot, reuse=False)
# placeholder
x_memory = tf.placeholder(tf.float32, shape=[None, memory_size, 28, 28, 1])     # memory_x
y_memory_ind = tf.placeholder(tf.int32, shape=[None, memory_size])
y_memory = tf.one_hot(y_memory_ind, ways)                                      # memory_label

x_classify = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])     # classify_x
y_classify_ind = tf.placeholder(tf.int32, shape=[None])
y_classify = tf.one_hot(y_classify_ind, ways)              # classify_label  转为one_hot

cnn = cnn_model.cnn()
x_target = cnn(x_classify, scope="target")        # classify特征值  [32, 64]
# # Removes dimensions of size 1 from the shape of a tensor.
# x_target = tf.squeeze(x_target, [1, 2])       # 将4维变为，因为2维 第2、3维度大小是1

memory_images = []
reuse = False
for i in range(memory_size):
    x_support = cnn(x_memory[:, i, :, :, :], scope="support", reuse=reuse)          # [32,64]
    # x_support = tf.squeeze(x_support, [1, 2])
    memory_images.append(x_support)
    reuse = True

# memory_images: [memory_size, 32 ,64]
##########################################################################################
# lstm_model_ just target with support  Demo1
if if_lstm:
    memory_images.append(x_target)
    lstm = BidirectionalLSTM(layer_sizes=[32], batch_size=batch_size)          # batch_size = 32  init
    out, _, _ = lstm(memory_images, name="lstm")
    memory_out = out[:-1]
    target_out = out[-1]
    # print("lstm", np.size(memory_out), target_out.shape)    # [5, 32, 64]  [32, 64]
else:
    memory_out = memory_images
    target_out = x_target

softmax_a, similarities = cd(memory_out, target_out)
#########################################################################################################

"""
##########################################################################################
# lstm_model_ support lstm and target another lstm  Demo2
if if_lstm:
    lstm = BidirectionalLSTM(layer_sizes=[32], batch_size=batch_size)          # batch_size = 32  init
    out, _, _ = lstm(memory_images, name="lstm")
    memory_out = out

    r_lstm = read_lstm(batch_size, 32,  feature_size)         # [5, 32, 64]
    softmax_a, similarities = r_lstm(memory_out, x_target, "r_lstm")
    print(softmax_a, similarities)
    # print("lstm", np.size(memory_out), target_out.shape)    # [5, 32, 64]  [32, 64]
else:
    memory_out = memory_images
    target_out = x_target
#########################################################################################################
"""

# 计算损失， optimizer模型
preds = tf.squeeze(tf.matmul(tf.expand_dims(softmax_a, 1), y_memory))         # [32]
# preds_one = tf.one_hot(tf.cast(preds,tf.int32), ways)
# print(y_memory)
top_k = tf.nn.in_top_k(preds, y_classify_ind, 1)
# [ True False False False  True False  True  True False False  True False
#  True False False  True  True False False False  True  True False False
#  True False  True  True False False  True  True]
# 这个函数用于在预测的时候判断预测结果是否正确。函数本来的意义为判断label(y_classify_ind)是不是在logits 的前k大的值，返回一个布尔值。
# 这个label是个index而不是向量.如果表示第4类为正确的，label=3 preds是一个向量
acc = tf.reduce_mean(tf.to_float(top_k))

correct_prob = tf.reduce_sum(tf.log(tf.clip_by_value(preds, 1e-10, 1.0))*y_classify, 1)
loss = tf.reduce_mean(-correct_prob, 0)

optim = tf.train.AdamOptimizer(learning_rate)
grads = optim.compute_gradients(loss)
train_step = optim.apply_gradients(grads)

# testing
test_acc = tf.reduce_mean(tf.to_float(top_k))

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
merged = tf.summary.merge_all()
test_summ = tf.summary.scalar('test avg accuracy', test_acc)
saver = tf.train.Saver()
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    step = 1
    time = np.zeros(training_iters)
    acc_show = np.zeros(training_iters)
    sess.run(tf.global_variables_initializer())
    with tqdm.tqdm(total=training_iters) as pbar:
        while step < training_iters:
            memory_x, memory_label, classify_x, classify_label = get_minibatch_train.get_minibatch()
            # print(memory_x.shape, memory_label.shape, classify_x.shape, classify_label.shape)
            # changing learning_rate
            if step < 1000:
                learning_rate = 1
            elif 1000 < step < 5000:
                learning_rate = 0.1
            elif 5000 < step < 10000:
                learning_rate = 0.01
            elif 10000 < step < 50000:
                learning_rate = 0.001
            else:
                learning_rate = 0.0001

            #  print(learning_rate)     注意需要sess.run的变量  _表示没用
            _, cost, _ = sess.run([train_step, loss, similarities], feed_dict={x_memory: memory_x,
                                             y_memory_ind: memory_label,
                                             x_classify: classify_x,
                                             y_classify_ind: classify_label
                                            })
            # print("cost ", cost)
            if step % display_step == 0:                   # 整除很好
                memory_x, memory_label, classify_x, classify_label = get_minibatch_test.get_minibatch()
                accauary, test_summary = sess.run([test_acc, test_summ], feed_dict={x_memory: memory_x, y_memory_ind: memory_label,
                                                             x_classify: classify_x, y_classify_ind: classify_label
                                                             })
                # print(step, "  acc:   ", accauary)
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()

                time[step] = step
                acc_show[step] = accauary
                # Tqdm 是一个快速，可扩展的Python进度条，可以在 python 长循环中添加一个进度提示信息，用户只需要封装任意的迭代器 tqdm(iterator)。
                pbar.set_description("train_accuracy: {}".format(accauary))
            pbar.update(1)
            step += 1
    weight_1 = sess.run(cnn.weight_1)
    weight_2 = sess.run(cnn.weight_2)
    weight_3 = sess.run(cnn.weight_3)
    weight_4 = sess.run(cnn.weight_4)
    biases_1 = sess.run(cnn.biases_1)
    biases_2 = sess.run(cnn.biases_2)
    biases_3 = sess.run(cnn.biases_3)
    biases_4 = sess.run(cnn.biases_4)
    offset_1 = sess.run(cnn.offset_1)
    offset_2 = sess.run(cnn.offset_2)
    offset_3 = sess.run(cnn.offset_3)
    offset_4 = sess.run(cnn.offset_4)
    scale_1 = sess.run(cnn.scale_1)
    scale_2 = sess.run(cnn.scale_2)
    scale_3 = sess.run(cnn.scale_3)
    scale_4 = sess.run(cnn.scale_4)

    np.savez("model\\weight_IMagnet_test.npz", weight1=weight_1, biases1=biases_1, offset1=offset_1,  scale1=scale_1,# npz文件读取用weight['arr_0']表示第一个数组
             weight2=weight_2, biases2=biases_2, offset2=offset_2,  scale2=scale_2, # 这里npz有10个数组,名字默认为arr_0 到 arr_9分别表示wc1到bout
             weight3=weight_3, biases3=biases_3, offset3=offset_3,  scale3=scale_3,  # 或者可以重命名weight1=wc1
             weight4=weight_4, biases4=biases_4, offset4=offset_4,  scale4=scale_4,
             )
    pbar.close()
    plt.plot(time, acc_show, "cs")
    plt.title('One-shot Learning')
    plt.xlabel('time')
    plt.ylabel('acc')
    plt.show()
    print(sum(acc_show[(training_iters-1000):])/1000)
