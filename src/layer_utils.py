import tensorflow as tf
import numpy as np
import itertools

from tensorflow.python.training import moving_averages


######## LAYERS ########
def dense(input_data, output_dim, name):
    input_dim = input_data.get_shape().as_list()[-1]
    """NN fully connected layer."""
    with tf.variable_scope(name):  
        W = tf.get_variable("W", [input_dim, output_dim],
                initializer=tf.contrib.layers.xavier_initializer())   
        b = tf.get_variable("b", [output_dim], initializer=tf.constant_initializer(0))
        return tf.matmul(input_data, W, name="matmul") + b

def batch_normalization(input_data, is_train, name='BatchNormalization'):
    """NN batch normalization layer."""
    x = input_data
    BN_DECAY = 0.9997
    BN_EPSILON = 0.001
    x_shape = x.get_shape()
    params_shape = x_shape[-1:]
    with tf.variable_scope(name):
      axis = list(range(len(x_shape) - 1))
      beta = tf.get_variable('beta',
          params_shape,
          initializer=tf.zeros_initializer)
      gamma = tf.get_variable('gamma',
          params_shape,
          initializer=tf.ones_initializer)
      moving_mean = tf.get_variable('moving_mean',
          params_shape,
          initializer=tf.zeros_initializer,
          trainable=False)
      moving_variance = tf.get_variable('moving_variance',
          params_shape,
          initializer=tf.ones_initializer,
          trainable=False)

      # These ops will only be preformed when training.
      mean, variance = tf.nn.moments(x, axis)
      update_moving_mean = moving_averages.assign_moving_average(moving_mean,
          mean, BN_DECAY)
      update_moving_variance = moving_averages.assign_moving_average(
          moving_variance, variance, BN_DECAY)
      tf.add_to_collection('update_ops', update_moving_mean)
      tf.add_to_collection('update_ops', update_moving_variance)

      mean, variance = tf.cond(
          is_train, lambda: (mean, variance),
          lambda: (moving_mean, moving_variance))

      x = tf.nn.batch_normalization(x, mean, variance, beta, gamma, BN_EPSILON)

    return x

def dense_relu_batch(input_data, N, H, is_train, name):
    """NN dense relu batch layer."""
    with tf.variable_scope(name):
        affine = dense(input_data, N, H, "dense")
        bn = batch_normalization(affine, is_train, "batch")
        return tf.nn.relu(bn, "relu")

def dense_relu(input_data, N, H, name):
    """NN dense relu layer"""
    with tf.variable_scope(name):
        affine = dense(input_data, N, H, "dense")
        return tf.nn.relu(affine, "relu")

def multi_dense_relu_batch(input_data, N, Hs, is_train, name):
    """NN multi dense relu batch layer."""
    with tf.variable_scope(name):
        output = input_data
        for i, H in enumerate(itertools.izip([N] + Hs, Hs)):
            output = dense_relu_batch(output, H[0], H[1], is_train, "fc_" + str(i))
        return output

def conv2d(input_data, filter_size, stride, name):
    """NN 2D convolutional layer."""
    with tf.variable_scope(name):
        W = tf.get_variable("W", filter_size,
                initializer=tf.contrib.layers.xavier_initializer_conv2d())
        conv = tf.nn.conv2d(input_data, W,
                [1, stride, stride, 1], "SAME", name="conv2d")
        biases = tf.get_variable("b", shape=filter_size[-1])
        bias = tf.reshape(tf.nn.bias_add(conv, biases),
                conv.get_shape().as_list())

        return bias

def conv_words(input_data, window_size, num_filters, name):
    """NN convolution over window_size words across entire embedding dimension"""
    batch_size, sentence_length, embedding_dim = input_data.get_shape().as_list()
    input_data = tf.reshape(input_data,
            [batch_size, sentence_length, embedding_dim, 1])
    with tf.variable_scope(name):
        filter_size = [window_size, embedding_dim, 1, num_filters]
        W = tf.get_variable("W", filter_size,
                initializer=tf.contrib.layers.xavier_initializer())
        conv = tf.nn.conv2d(input_data, W, [1,1,1,1], padding='VALID')
        biases = tf.get_variable("b", shape=filter_size[-1])
        bias = tf.reshape(tf.nn.bias_add(conv, biases),
                conv.get_shape().as_list())
        return bias

def maxpool2d(input_data, stride, name):
    """NN 2D max pooling layer."""
    with tf.variable_scope(name):
        filter_size = [1, stride, stride, 1]
        return tf.nn.max_pool(input_data, filter_size,
                filter_size, "SAME", name="max_pool")

def conv2d_relu_batch(input_data, filter_size, stride, is_train, name):
    with tf.variable_scope(name):
        conv = conv2d(input_data, filter_size, stride, "conv2d")
        bn = batch_normalization(conv, is_train, "batch")
        return tf.nn.relu(bn, "relu")

def conv2d_relu(input_data, filter_size, stride, name):
    with tf.variable_scope(name):
        conv = conv2d(input_data, filter_size, stride, "conv2d")
        return tf.nn.relu(conv, "relu")

def softmax_loss(logits, labels):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits,
            labels, name='cross_entropy')
    cross_entropy_mean = tf.reduce_mean(
        cross_entropy, name='mean_cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')
