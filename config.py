__author__ = "Nova Mind"
__date__ = "$2018-7-13"

"""
   data: UISEE datasets

"""
import os
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS


#######################
# Training Parameters #
#######################
tf.app.flags.DEFINE_float('weight_decay', 0.0001, "Weight decay, for regularization")
tf.app.flags.DEFINE_float('epsilon', 1e-4, "epsilon for BN")
tf.app.flags.DEFINE_float('learning_rate', 0.001, "Learning rate")
tf.app.flags.DEFINE_float('momentum', 0.9, "Momentum")
tf.app.flags.DEFINE_float('gamma', 0.1, "Factor for reducing the learning rate")

tf.app.flags.DEFINE_integer('epoch', 500, 'how many epoch for training')
tf.app.flags.DEFINE_integer('batch_size', 32, "Network batch size:cifar10 :128; ImageNet:")
tf.app.flags.DEFINE_integer('iteration_numbers', 2000, "iteration number for train")
tf.app.flags.DEFINE_integer('test_iter_num', 5, "iteration number for test")

tf.app.flags.DEFINE_integer('saver_step', 200,
                            "the step of save the model")
tf.app.flags.DEFINE_integer('display_step', 10,
                            "Iteration intervals for showing the loss during training, on command line interface")
tf.app.flags.DEFINE_integer('ways', 5, "5-ways 5 classes for match")
tf.app.flags.DEFINE_integer('shot', 5, "1-shot each ways only 1 image for memory")

tf.app.flags.DEFINE_integer('input_image', [28, 28, 1], "input image size gray image")
tf.app.flags.DEFINE_integer('feature_size', 64, "output of cnn and lstm")
#######################
#   DATA  Parameters  #
#######################
tf.app.flags.DEFINE_string('train_data', 'data/images_background', "train data path")
tf.app.flags.DEFINE_string('test_data', 'data/images_evaluation', 'test data path')
tf.app.flags.DEFINE_float('data_mean', 234, 'mean of data')
tf.app.flags.DEFINE_float('data_std', 68, 'std of data')


# ------------------------------------------------------------------------------
