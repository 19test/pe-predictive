import unittest
import numpy as np
import tensorflow as tf

from os.path import join, dirname
from models import *
from classifier import *
from shutil import rmtree

class TestLayers(unittest.TestCase):
    def setUp(self):
        tf.reset_default_graph()

    def test_word_model(self):
        opts = WordCNNOpts('test')
        vocab_size = 10
        embedding_dim = 100
        embedding_np = np.random.rand(vocab_size, embedding_dim)
        model = CNN_Word_Model(opts, embedding_np)
        batchX = np.random.randint(vocab_size,
                size=(opts.batch_size, opts.sentence_len))
        batchy = np.random.randint(2, size=opts.batch_size)
        train_loss, train_acc = model.step(batchX, batchy, train=True)
        val_loss, val_acc = model.step(batchX, batchy, train=False)
        assert not np.isnan(train_loss)
        assert 0.0 <= train_acc <= 1.0
        assert not np.isnan(val_loss)
        assert 0.0 <= val_acc <= 1.0

        pred = model.predict(batchX)
        assert np.array([p in [0,1] for p in pred]).all()
        model.save_weights(100)
        model.restore_weights()
        rmtree(opts.archlog_dir)

if __name__ == '__main__':
    unittest.main()
