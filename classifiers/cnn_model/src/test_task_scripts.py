import unittest
import numpy as np

from ModelFactory import GlobalOpts
from reader import BatchIterator

from task1 import train as task1_train
from task1 import test as task1_test
from task1 import predict as task1_predict

#from task2 import train as task2_train
#from task2 import test as task2_test


class TestModel:
    def __init__(self, opts, num_outputs):
        self.opts = opts
        self.num_outputs = num_outputs

    def step(self, batchX, batchy, train):
        return 10.5, 0.5

    def save_weights(self, it):
        pass

    def restore_weights(self):
        pass

    def predict(self, batchX):
        return np.ones((self.opts.batch_size, self.num_outputs))

class TestReader:
    def __init__(self, opts, num_outputs):
        self.opts = opts
        self.num_outputs = num_outputs
        self.num_examples = 1000
        self.setX = np.random.randint(1000,size=(self.num_examples,
            self.opts.sentence_len))
        self.sety = np.vstack((np.ones((self.num_examples/2,num_outputs)),
            np.zeros((self.num_examples/2,num_outputs))))

    def _sample(self):
        inds = np.random.choice(range(self.setX.shape[0]),
                replace=False, size=self.opts.batch_size)
        return self.setX[inds,:], self.sety[inds,:]

    def sample_train(self):
        return self._sample()

    def sample_val(self):
        return self._sample()

    def get_test_batches(self):
        return BatchIterator(self.setX, self.sety,
                self.opts.batch_size, self.opts.sentence_len)

class TestLogger:
    def log(self, data):
        pass

class TestTask1(unittest.TestCase):
    def setUp(self):
        self.opts = GlobalOpts('test_task1')
        self.opts.error_analysis = False
        self.model = TestModel(self.opts, 1)
        self.reader = TestReader(self.opts, 1)
        self.logger = TestLogger()

    def test_train_function(self):
        task1_train(self.model, self.reader, self.logger, self.opts)

    def test_test_function(self):
        acc, prec, recall = task1_test(self.model, self.reader,
                self.logger, self.opts)
        assert acc == 0.5, acc
        assert prec == 0.5, prec
        assert recall == 1.0, recall

    def test_predict_function(self):
        input_batches = self.reader.get_test_batches()
        output = task1_predict(input_batches, self.model, self.opts)
        assert output.shape == (self.reader.num_examples,
                self.reader.num_outputs)
        assert (output == 1).all()

'''
class TestTask2(unittest.TestCase):
    def setUp(self):
        assert False

    def test_train_function(self):
        assert False

    def test_test_function(self):
        assert False
'''

if __name__ == '__main__':
    unittest.main()
