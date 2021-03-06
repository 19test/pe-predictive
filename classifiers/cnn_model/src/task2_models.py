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
        self.y_in = tf.placeholder(tf.int64, shape=(self.opts.batch_size, 2))
        self.y_pe, self.y_burden = tf.unpack(self.y_in, axis=1)
        self.keep_prob = tf.placeholder(tf.float32)
        self.global_step = tf.Variable(0, trainable=False)

        # implementation left to different model classes
        drop = self.create_graph()
        self.logits_pe = dense(drop, 2, name='dense_pe')
        self.logits_burden = dense(drop, 5, name='dense_burden')

        # Generalized optimization steps and metrics
        self.loss_pe = softmax_loss(self.logits_pe, self.y_pe)
        self.loss_burden = softmax_loss(self.logits_burden, self.y_burden)
        self.loss = self.loss_pe + self.loss_burden

        self.acc_pe = tf.reduce_mean(tf.cast(tf.equal(
            tf.argmax(self.logits_pe, 1),self.y_pe), tf.float32))
        self.acc_burden = tf.reduce_mean(tf.cast(tf.equal(
            tf.argmax(self.logits_burden, 1),self.y_burden), tf.float32))

        self.pred_pe = tf.nn.softmax(self.logits_pe)
        self.pred_burden = tf.nn.softmax(self.logits_burden)
        
        self.train_op = self._train_op_init(self.loss,
                global_step=self.global_step)
        restore_variables = tf.trainable_variables() \
                + tf.moving_average_variables()
        self.saver = tf.train.Saver(restore_variables)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())


    #TODO : apply gradient clipping
    def _train_op_init(self, total_loss, global_step):
        INITIAL_LEARNING_RATE = self.opts.init_lr
        LEARNING_RATE_DECAY_FACTOR = self.opts.decay_factor
        DECAY_STEPS = self.opts.decay_steps

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
        step_ops.append(self.acc_pe)
        step_ops.append(self.acc_burden)
        if train:
            step_ops += self.train_op
            keep_prob = self.opts.keep_prob
        else:
            keep_prob = 1.0
        result = self.sess.run(step_ops,
                feed_dict={
                    self.x_in:batchX,
                    self.y_in:batchy,
                    self.keep_prob:keep_prob
                    })
        return result[0], [result[1], result[2]]

    def predict(self, batchX):
        scores = self.sess.run([self.pred_pe, self.pred_burden],
                feed_dict={
                    self.x_in:batchX,
                    self.y_in:np.random.randint(2, size=(self.opts.batch_size, 2)),
                    self.keep_prob:1.0
                    })
        prediction_pe = np.argmax(scores[0], axis=1).reshape(
                self.opts.batch_size,1)
        prediction_burden = np.argmax(scores[1], axis=1).reshape(
                self.opts.batch_size,1)
        return np.hstack((prediction_pe, prediction_burden))

class LSTM_Model(Model):

    def __init__(self, opts, embedding_np):
        self.embedding_np = embedding_np
        super(LSTM_Model, self).__init__(opts)

    def create_graph(self):

        embedding = tf.get_variable(name="W", shape=self.embedding_np.shape,
                initializer=tf.constant_initializer(self.embedding_np),
                trainable=False)
        del self.embedding_np
        word_vecs = tf.nn.embedding_lookup(embedding, self.x_in)
        
        # input_data : (batch_size, sentence_len, vec_len)
        input_data = tf.split(1, self.opts.sentence_len, word_vecs)
        with tf.variable_scope('lstm') as scope:
            multi_lstm = tf.nn.rnn_cell.MultiRNNCell(
                    [tf.nn.rnn_cell.BasicLSTMCell(self.opts.hidden_size)] * \
                            self.opts.num_layers)
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
                initializer=tf.constant_initializer(self.embedding_np), 
                trainable=False)
        del self.embedding_np 
        word_vecs = tf.nn.embedding_lookup(embedding, self.x_in)
        conv = conv_words(word_vecs, self.opts.window_size, 
                self.opts.num_filters, 'conv')
        relu = tf.nn.relu(conv, name='relu')
        maxpool = tf.squeeze(tf.reduce_max(relu, 1, name='maxpool'))
        # consider replacing dropout with batch norm in future with more data
        drop = tf.nn.dropout(maxpool, self.keep_prob)
        return drop


class CNN_Char_Model(Model):

    def create_graph(self):
        return 0
