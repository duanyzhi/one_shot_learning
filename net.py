from config import FLAGS as cfg
from load_data import data
from cnn_model import cnn_model
from lstm_model import BidirectionalLSTM
from cosine_distance import cd
from utils import *
import tensorflow as tf


data = data()
cnn  =cnn_model()

class network(object):
    def __init__(self, pattern):
        self.pattern = pattern
        self.placeholder()
        self.if_lstm = False  # default is do not use lstm if not good
        self.Parameters = {}
        pass

    def placeholder(self):
        self.memory_x = tf.placeholder(tf.float32, shape=(None, cfg.ways*cfg.shot, cfg.input_image[0], cfg.input_image[1], cfg.input_image[2]))
        self.memory_label = tf.placeholder(tf.int32, shape=(None,  cfg.ways*cfg.shot))
        self.y_memory = tf.one_hot(self.memory_label, cfg.ways)
        self.classify_x = tf.placeholder(tf.float32, shape=(None, cfg.input_image[0], cfg.input_image[1], cfg.input_image[2]))   # target
        self.classify_label = tf.placeholder(tf.int32, shape=(None))
        self.y_classify = tf.one_hot(self.classify_label, cfg.ways)
        self.is_training = tf.placeholder(tf.bool)

        pass

    def build_net(self):
        x_target = cnn(self.classify_x, self.is_training)
        memory_images = []
        for i in range(cfg.ways*cfg.shot):
            x_support = cnn(self.memory_x[:, i, :, :, :], self.is_training)          # [32,64]
            # x_support = tf.squeeze(x_support, [1, 2])
            memory_images.append(x_support)

        # memory_images: [memory_size, 32 ,64]
        ##########################################################################################
        # lstm_model_ just target with support  Demo1
        if self.if_lstm:
            memory_images.append(x_target)
            lstm = BidirectionalLSTM(layer_sizes=[32], batch_size=cfg.batch_size)          # batch_size = 32  init
            out, _, _ = lstm(memory_images, name="lstm")
            memory_out = out[:-1]
            target_out = out[-1]
        else:
            memory_out = memory_images
            target_out = x_target

        softmax_a, similarities = cd(memory_out, target_out)
        self.Parameters["soft"] = softmax_a
        if self.pattern  == "train":
            self.BP()
        pass

    def BP(self):
        # 计算损失， optimizer模型
        preds = tf.squeeze(tf.matmul(tf.expand_dims(self.Parameters["soft"], 1), self.y_memory))         # [32]
        # preds_one = tf.one_hot(tf.cast(preds,tf.int32), ways)
        # print(y_memory)
        top_k = tf.nn.in_top_k(preds, self.classify_label, 1)
        # [ True False False False  True False  True  True False False  True False
        #  True False False  True  True False False False  True  True False False
        #  True False  True  True False False  True  True]
        # 这个函数用于在预测的时候判断预测结果是否正确。函数本来的意义为判断label(self.classify_label)是不是在logits 的前k大的值，返回一个布尔值。
        # 这个label是个index而不是向量.如果表示第4类为正确的，label=3 preds是一个向量
        acc = tf.reduce_mean(tf.to_float(top_k))

        correct_prob = tf.reduce_sum(tf.log(tf.clip_by_value(preds, 1e-10, 1.0))*self.y_classify, 1)
        loss = tf.reduce_mean(-correct_prob, 0)

        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        optim = tf.train.AdamOptimizer(cfg.learning_rate)
        grads = optim.compute_gradients(loss)
        train_step = optim.apply_gradients(grads)

        self.Parameters["loss"] = loss
        self.Parameters["acc"] = acc
        self.Parameters["euo"] = extra_update_ops
        self.Parameters["train_step"] = train_step


    def gpu(self):
        saver = tf.train.Saver()
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.8
        session = tf.Session(config=config)

        # init unsaved variable when use ckpt
        # session.run(tf.local_variables_initializer())

        # print("Loading ckpt Model, have a coffee...")
        saver.restore(session, 'data/ckpt/five_shot_1000.ckpt')
        # print("Succeed Load ckpt!")

        # init variable when traning without ckpt
        # session.run(tf.global_variables_initializer())
        return session, saver

    def run(self):
        sess, saver = self.gpu()
        if self.pattern == "train":
            loss_list = [["iter", "loss"]]
            acc_list_train = [["iter", "acc"]]
            acc_list_test = [["iter", "acc"]]
            for kk in range(1001, cfg.iteration_numbers):
                lr(kk)
                memory_data, memory_label, target_data, target_label = data("train")
                feed_dict = {self.memory_x:memory_data, self.memory_label:memory_label,
                             self.classify_x:target_data, self.classify_label:target_label,
                             self.is_training: True}
                out = sess.run(self.Parameters, feed_dict=feed_dict)
                if kk % cfg.display_step == 0:
                    acc = self.run_test(sess)
                    print("iter:%05d"%kk, "Loss:%.3f"%out["loss"], "Train Acc:%.3f"%out["acc"], "Test Acc:%.3f"%acc)
                    loss_list.append([kk, out["loss"]])
                    acc_list_train.append([kk, out["acc"]])
                    acc_list_test.append([kk, acc])
                if kk % cfg.saver_step == 0:
                    save_csv('data/csv/loss_new.csv', loss_list)
                    save_csv('data/csv/train_acc_new.csv', acc_list_train)
                    save_csv('data/csv/test_acc_new.csv', acc_list_test)
                    saver.save(sess, 'data/ckpt/five_shot_' + str(kk) + '.ckpt')
        else:
            acc = self.run_test(sess)

    def run_test(self, sess):
        # random is not very good
        test_acc = []
        for kk in range(100):
            memory_data, memory_label, target_data, target_label = data("test")
            feed_dict = {self.memory_x:memory_data, self.memory_label:memory_label,
                         self.classify_x:target_data, self.classify_label:target_label,
                         self.is_training: False}
            test_out = sess.run(self.Parameters["acc"], feed_dict=feed_dict)
            test_acc.append(test_out)
        return sum(test_acc)/len(test_acc)



# ------------------------------------------------------------------------------
