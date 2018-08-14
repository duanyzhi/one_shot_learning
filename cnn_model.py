import tensorflow as tf


def residual_block(input_img, filter_size, scope_name, model):
    input_depth = int(input_img.get_shape()[3])
    with tf.variable_scope(scope_name):
        conv1 = tf.layers.conv2d(inputs=input_img, filters=filter_size, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                 activation=tf.nn.relu, name='conv1')
        conv2 = tf.layers.conv2d(inputs=conv1, filters=filter_size, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                 name='conv2')
        bn_3 = tf.layers.batch_normalization(conv2, name='bn', training=model)
        padding_zeros = tf.pad(input_img, [[0, 0], [0, 0], [0, 0], [int((filter_size - input_depth) / 2),
                                                                    filter_size - input_depth - int(
                                                                   (filter_size - input_depth) / 2)]])
    res_block = padding_zeros + bn_3
    return res_block

# you can use very simple cnn model 
class cnn_model(object):
    def __init__(self):
        self.reuse = False

    def __call__(self, features, is_training):
        with tf.variable_scope("cnn") as scope_name:
            if self.reuse:
                scope_name.reuse_variables()
            # conv2
            bn_1 = tf.layers.batch_normalization(features, name='bn1', training=is_training)
            res_block_1 = residual_block(bn_1, 64, "res_block_1", is_training)
            res_block_2 = residual_block(res_block_1, 64, "res_block_2", is_training)
            conv2 = residual_block(res_block_2, 64, "conv2", is_training)
            pool2 = tf.layers.max_pooling2d(conv2, [2, 2], 2)
            # conv3
            res_block_3 = residual_block(pool2, 64, "res_block_3", is_training)
            res_block_4 = residual_block(res_block_3, 64, "res_block_4", is_training)
            res_block_5 = residual_block(res_block_4, 64, "res_block_5", is_training)
            conv3 = residual_block(res_block_5, 64, "conv3", is_training)
            pool3 = tf.layers.max_pooling2d(conv3, [2, 2], 2)
            # conv4
            res_block_6 = residual_block(pool3, 64, "res_block_6", is_training)
            res_block_7 = residual_block(res_block_6, 64, "res_block_7", is_training)
            res_block_8 = residual_block(res_block_7, 64, "res_block_8", is_training)
            res_block_9 = residual_block(res_block_8, 64, "res_block_9", is_training)
            res_block_10 = residual_block(res_block_9, 64, "res_block_10", is_training)
            conv4 = residual_block(res_block_10, 64, "conv4", is_training)
            pool4 = tf.layers.max_pooling2d(conv4, [2, 2], 2)
            # conv5
            res_block_11 = residual_block(pool4, 64, "res_block_11", is_training)
            res_block_12 = residual_block(res_block_11, 64, "res_block_12", is_training)
            conv5 = residual_block(res_block_12, 64, "conv5", is_training)
            pool5 = tf.layers.max_pooling2d(conv5, [2, 2], 2)
            fc_ave_pool =  tf.squeeze(pool5, [1, 2])
            out = tf.layers.batch_normalization(fc_ave_pool,  name='bn_fc2', training=is_training)
        self.reuse = True
        return out   # [batch, 64]


# ----------------------------------------------------END----------------------------------------------------
