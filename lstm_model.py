import tensorflow as tf
import tensorflow.contrib.rnn as rnn
from cosine_distance import compute_distance

class BidirectionalLSTM:
    def __init__(self, layer_sizes, batch_size):
        self.reuse = False
        self.batch_size = batch_size
        self.layer_sizes = layer_sizes

    def __call__(self, inputs, name):
        """"
         只要定义类型的时候，实现__call__函数，这个类型就成为可调用的,相当于重载了括号运算符
         lstm = BidirectionalLSTM():执行__init__
         lstm():再次调用()里是(self, inputs, name, training=False)：执行__call__
        """
        with tf.name_scope('bid_lstm' + name), tf.variable_scope('bid_lstm', reuse=self.reuse):
            fw_lstm_cells = [rnn.LSTMCell(num_units=self.layer_sizes[i], activation=tf.nn.tanh)
                             for i in range(len(self.layer_sizes))]
            bw_lstm_cells = [rnn.LSTMCell(num_units=self.layer_sizes[i], activation=tf.nn.tanh)
                             for i in range(len(self.layer_sizes))]

            # 双向LSTM outputs是最后相加前向反向的输出  fw：前向lstm输出 bw反向lstm输出
            outputs, output_state_fw, output_state_bw = rnn.stack_bidirectional_rnn(
                fw_lstm_cells,
                bw_lstm_cells,
                inputs,
                dtype=tf.float32
            )

        self.reuse = True  # 共享lstm
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='bid-lstm')
        # print(outputs.shape, output_state_fw.shape, output_state_bw.shape)
        return outputs, output_state_fw, output_state_bw


class read_lstm:
    def __init__(self, layer_r_sizes, batch_r_size, feature_size):
        self.reuse = False
        self.batch_r_size = batch_r_size
        self.layer_r_sizes = layer_r_sizes
        self.feature_size = feature_size

    def __call__(self, memory_out, x_target, name):                                                 # [5, 32, 64]
        # calculate h1
        lstm_cell_s = rnn.LSTMCell(self.feature_size, input_size=self.feature_size, forget_bias=1.0, reuse=False)
        state = lstm_cell_s.zero_state(self.batch_r_size, dtype=tf.float32)
        outputs, state = lstm_cell_s(x_target, state, scope="test")
        (c_pre, h_pre) = state
        h_nex = h_pre + x_target
        # cosine distance
        _, _, r_pre = compute_distance(memory_out, h_nex)   # r_pre [32, 64]

        # for simple Let r be in x_target
        h_pre = tf.concat([h_nex, r_pre], 1)           # h_nex [32, 128]

        new_state = (rnn.LSTMStateTuple(c_pre, h_pre))

        lstm_cell = rnn.LSTMCell(self.feature_size, input_size=self.feature_size, forget_bias=1.0, reuse=False)
        state = new_state
        outputs, state = lstm_cell(x_target, state, scope=name)  # [32, 64]
        # print(outputs, state)
        (c_pre, h_pre) = state
        h_nex = h_pre + x_target
        softmax_a, similarities, r_pre = compute_distance(memory_out, h_nex)
        h_pre = tf.concat([h_nex, r_pre], 1)
        new_state = (rnn.LSTMStateTuple(c_pre, h_pre))
        state = new_state
        for kk in range(1, self.layer_r_sizes):
            # print(kk)
            lstm_cell._reuse = True
            outputs, state = lstm_cell(x_target, state, scope=name)       # input  [32, 64]:batch_size*feature_size
            (c_pre, h_pre) = state
            h_nex = h_pre + x_target
            softmax_a, similarities, r_pre = compute_distance(memory_out, h_nex)
            h_pre = tf.concat([h_nex, r_pre], 1)
            new_state = (rnn.LSTMStateTuple(c_pre, h_pre))
            state = new_state
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)
        return softmax_a, similarities
