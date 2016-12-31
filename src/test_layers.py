import unittest
import numpy as np
import tensorflow as tf

from os.path import join, dirname
from layer_utils import *

class TestReader(unittest.TestCase):
    def setUp(self):
        tf.reset_default_graph()

    def test_conv_words(self):
        batch_size = 20
        sentence_len = 30
        embedding_dim = 40
        window_size = 5
        num_filters = 10
        name = 'test'

        def conv_words_impl(input_data, weights, window_size, num_filters):
            batch_size, sentence_length, embedding_dim = input_data.shape
            assert weights.shape == (window_size,embedding_dim,1,num_filters)
            result = np.zeros((batch_size,
                sentence_length - window_size + 1, 1, num_filters))
            for filter_ind in range(weights.shape[-1]):
                for i in range(batch_size):
                    for j in range(sentence_length - window_size + 1):
                        window = input_data[i,j:j+window_size,:]
                        filter_weights = np.squeeze(weights[:,:,:,filter_ind])
                        result[i,j,0,filter_ind] = np.sum(window * filter_weights)
            return result

        example = np.random.rand(batch_size, sentence_len,
                embedding_dim).astype(np.float32)
        input_data = tf.constant(example)
        conv = conv_words(input_data, window_size=window_size,
                num_filters=num_filters, name=name)
        with tf.variable_scope(name, reuse=True):
            weights_var = tf.get_variable("W")
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            weights = sess.run(weights_var)
            result = sess.run(conv)
        result_comp = conv_words_impl(example, weights,
                window_size, num_filters)
        assert result.shape == result_comp.shape,\
                (result.shape, result_comp.shape)
        assert (result - result_comp < 10e-4).all()


if __name__ == '__main__':
    unittest.main()
