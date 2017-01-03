import pandas as pd
import numpy as np
import tensorflow as tf
import os

from os.path import join, dirname
from layer_utils import *


### Models

class Model(object):
    def __init__(self,  opts):
        self.opts = opts
        # Model input variables
        self.x_in = tf.placeholder(tf.int64,
                shape=(self.opts.batch_size, self.opts.sentence_len))
        self.y_in = tf.placeholder(tf.int64, shape=(self.opts.batch_size))
        self.keep_prob = tf.placeholder(tf.float32)
        self.global_step = tf.Variable(0, trainable=False)

        # implementation left to different model classes
        self.logits = self.create_graph()

        # Generalized optimization steps and metrics
        self.loss = softmax_loss(self.logits, self.y_in)
        self.acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.logits, 1),
            self.y_in), tf.float32))
        self.pred = tf.nn.softmax(self.logits)
        self.train_op = self._train_op_init(self.loss, global_step=self.global_step)
        restore_variables = tf.trainable_variables() \
                + tf.moving_average_variables()
        self.saver = tf.train.Saver(restore_variables)

        self.sess = tf.Session()
        self.sess.run(tf.initialize_all_variables())


    #TODO : apply gradient clipping
    def _train_op_init(self, total_loss, global_step):
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

    def create_graph(self):
        raise Exception('Abstract class - Override')

    def save_weights(self, it):
        if not os.path.exists(self.opts.archlog_dir):
            os.makedirs(self.opts.archlog_dir)
        checkpoint_path = join(self.opts.archlog_dir, 'checkpoint.ckpt')
        self.saver.save(self.sess, checkpoint_path, global_step=it)

    def restore_weights(self):
        all_ckpts = tf.train.get_checkpoint_state(self.opts.archlog_dir)
        self.saver.restore(self.sess, all_ckpts.model_checkpoint_path)

    def step(self, batchX, batchy, train=True):
        '''
        Run one training step of SGD 
        Returns : [loss, acc, rmse] of batch
        '''
        step_ops = []
        step_ops.append(self.loss)
        step_ops.append(self.acc)
        if train:
            step_ops += self.train_op
            keep_prob = self.opts.keep_prob
        else:
            keep_prob = 1.0
        result = self.sess.run(step_ops, feed_dict={self.x_in:batchX,
            self.y_in:batchy, self.keep_prob:keep_prob})
        return result[0], result[1]

    def predict(self, batchX):
        scores = self.sess.run(self.pred, feed_dict={self.x_in:batchX,
            self.y_in:np.random.rand(self.opts.batch_size)})
        prediction = np.argmax(scores, axis=1)
        return prediction

class LSTM_Model(Model):

    def __init__(self, opts, embedding_np):
        self.embedding_np = embedding_np
        super(LSTM_Model, self).__init__(opts)

    def create_graph(self):

        embedding = tf.get_variable(name="W", shape=self.embedding_np.shape,
                initializer=tf.constant_initializer(self.embedding_np), trainable=False)
        del self.embedding_np
        word_vecs = tf.nn.embedding_lookup(embedding, self.x_in)
        
        # input_data : (batch_size, sentence_len, vec_len)
        input_data = tf.split(1, self.opts.sentence_len, word_vecs)
        with tf.variable_scope('lstm') as scope:
            multi_lstm = tf.nn.rnn_cell.MultiRNNCell(
                    [tf.nn.rnn_cell.BasicLSTMCell(self.opts.hidden_size)] * self.opts.num_layers)
            state = multi_lstm.zero_state(self.opts.batch_size, tf.float32)
            for t in range(self.opts.sentence_len):
                output, state = multi_lstm(tf.squeeze(input_data[t]), state)
                scope.reuse_variables()
        logits = dense(output, 2, name='dense')
        return logits

class CNN_Word_Model(Model):

    def __init__(self, opts, embedding_np):
        self.embedding_np = embedding_np
        super(CNN_Word_Model, self).__init__(opts)

    def create_graph(self):

        embedding = tf.get_variable(name="W", shape=self.embedding_np.shape,
                initializer=tf.constant_initializer(self.embedding_np), trainable=False)
        del self.embedding_np 
        word_vecs = tf.nn.embedding_lookup(embedding, self.x_in)
        conv = conv_words(word_vecs, self.opts.window_size, self.opts.num_filters, 'conv')
        relu = tf.nn.relu(conv, name='relu')
        maxpool = tf.squeeze(tf.reduce_max(relu, 1, name='maxpool'))
        # consider replacing dropout with batch norm in future with more data
        drop = tf.nn.dropout(maxpool, self.keep_prob)
        logits = dense(drop, 2, name='dense')
        return logits


class CNN_Char_Model(Model):

    def create_graph(self):
        return 0
