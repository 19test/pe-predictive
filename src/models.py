import pandas as pd
import numpy as np
import tensorflow as tf

from os.path import join, dirname

#TODO : apply gradient clipping
def _train_op_init(total_loss, global_step):
    INITIAL_LEARNING_RATE = 0.001
    LEARNING_RATE_DECAY_FACTOR = 0.1
    DECAY_STEPS = 5000

    self.lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                    global_step,
                                    DECAY_STEPS,
                                    LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)
    opt = tf.train.AdamOptimizer(self.lr)
    grads = opt.compute_gradients(total_loss)
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
    train_ops = [apply_gradient_op]
    train_ops += tf.get_collection('update_ops')
    return train_ops

# Compute softmax cross entropy loss
def softmax_loss(logits, labels):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits,
            labels, name='cross_entropy')
    cross_entropy_mean = tf.reduce_mean(
        cross_entropy, name='mean_cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def dense(input_data, N, H, name):
    """NN fully connected layer."""
    with tf.variable_scope(name):  
        W = tf.get_variable("W", [N, H],
                initializer=tf.contrib.layers.xavier_initializer())   
        b = tf.get_variable("b", [H], initializer=tf.constant_initializer(0))
        return tf.matmul(input_data, W, name="matmul") + b

def dense_relu(input_data, output_dim, name="dense_relu"):
    """NN dense relu layer"""
    batch_size, in_dim = input_data.get_shape()
    with tf.variable_scope(name):
        affine = dense(input_data, in_dim, output_dim, "dense")
        return tf.nn.relu(affine, "relu")


### Models

class Model:
    def __init__(self,  opts):
        self.opts = opts
        # Model input variables
        self.x_in = tf.Placeholder(tf.float32, shape=(batch_size, sentence_len))
        self.y_in = tf.Placeholder(tf.int64, shape=(batch_size))
        self.global_step = tf.Variable(0, trainable=False)

        # implementation left to different model classes
        logits = self.create_graph()

        # Generalized optimization steps and metrics
        logits = dense_relu(logits, output_dim=2)
        loss = softmax_loss(logits, gt_labels)
        acc = tf.reduce_mean(tf.equal(tf.argmax(logits, 1), y_in))
        pred = tf.nn.softmax(logits)
        train_op = _train_op_init(loss, global_step=global_step)
        restore_variables = tf.trainable_variables() \
                + tf.moving_average_variables()
        self.saver = tf.train.Saver(restore_variables)

        self.sess = tf.Session()
        self.sess.run(tf.initialize_all_variables())

    def save_weights(self):
        if not os.path.exists(self.opts.ARCHLOG_DIR):
            os.makedirs(self.opts.ARCHLOG_DIR)
        checkpoint_path = join(self.opts.ARCHLOG_DIR, 'checkpoint.ckpt')
        self.saver.save(self.sess, checkpoint_path, global_step=it)

    def restore_weights(self):
        all_ckpts = tf.train.get_checkpoint_state(self.opts.ARCHLOG_DIR)
        self.saver.restore(self.sess, all_ckpts.model_checkpoint_path)

    def step(self, batchX, batchy, train=True):
        '''
        Run one training step of SGD 
        Returns : [loss, acc, rmse] of batch
        '''
        step_ops = []
        step_ops.append(self.loss)
        step_ops.append(self.accuracy)
        if train:
          step_ops += self.train_op
        result = self.sess.run(step_ops,
                feed_dict={self.x_in:batchX, self.y_in:batchy})
        return result[0], result[1]

    def pred(self, batchX):
        return 0

class LSTM_Model(Model):

    def __init__(self, opts, embedding_np):
        self.embedding_np = embedding_np
        super(opts)

    def create_graph(self):
        embedding_np = self.embedding_np
        del self.embedding_np

        embedding = tf.get_variable(name="W", shape=embedding_np.shape,
                initializer=tf.constant_initializer(embedding_np), trainable=False)
        word_vecs = tf.nn.embedding_lookup(embedding, x_in)
        

        embedding_len = sampler.get_embedding_len()
        logits = lstm(x_in, hiddens_size=hidden_size, num_layers=num_layers)

        # input_data : (batch_size, sentence_len, vec_len)
        batch_size, sentence_len, embedding_len = input_data.get_shape()
        input_data = tf.split(1, sentence_len, x_in)
        with tf.variable_scope(name) as scope:
            multi_lstm = tf.nn.rnn_cell.MultiRNNCell(
                    [tf.nn.rnn_cell.BasicLSTMCell(hidden_size)] * num_layers)
            state = multi_lstm.zero_state(batch_size, tf.float32)
            for t in range(sentence_len):
                output, state = multi_lstm(input_data[t], state)
                scope.reuse_variables()
        return output

class CNN_Word_Model(Model):

    def create_graph(self):
        return 0

class CNN_Char_Model(Model):

    def create_graph(self):
        return 0
