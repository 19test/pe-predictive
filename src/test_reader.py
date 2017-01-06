from classifier import GlobalOpts

import unittest
import numpy as np
from os.path import join, dirname
from reader import Reader

class TestReader(unittest.TestCase):
    def setUp(self):
        self.opts = GlobalOpts('test')
        self.data_paths = [join(self.opts.project_dir, 'data', 'stanford_pe.tsv')]
        self.reader = Reader(opts=self.opts, data_paths=self.data_paths)

    def test_split_train(self):
        train_size = len(self.reader.trainX)
        val_size = len(self.reader.valX)
        test_size = len(self.reader.testX)
        df_size = pd.read_csv(self.data_paths[0],sep='\t').shape[0]
        print 'Sizes - Train : %d Val : %d Test : %d' % (train_size, val_size, test_size)
        assert df_size == train_size + val_size + test_size 

    def test_get_embedding(self):
        embedding_np = self.reader.get_embedding(self.opts.glove_path)
        with open(self.opts.glove_path, 'r') as f:
            for line in f:
                data = line.split(' ')
                word = data[0]
                if word in self.reader.word_to_id:
                    word_vec = np.array(data[1::]).astype(np.float32)
                    word_id = self.reader.word_to_id[word]
                    embedding_vec = embedding_np[word_id,:]
                    assert word_vec.shape == embedding_vec.shape
                    assert (embedding_np[word_id,:] == word_vec).all(),\
                            (np.sum(embedding_np[word_id,:]), np.sum(word_vec))

    def test_sample_train(self):
        batchX, batchy = self.reader.sample_train()
        assert batchX.shape == (self.opts.batch_size, self.opts.sentence_len)
        assert batchy.shape == (self.opts.batch_size,)
        assert np.min(batchy) >= 0 and np.max(batchy) <= 1

    def test_sample_val(self):
        batchX, batchy = self.reader.sample_val()
        assert batchX.shape == (self.opts.batch_size, self.opts.sentence_len)
        assert batchy.shape == (self.opts.batch_size,)
        assert np.min(batchy) >= 0 and np.max(batchy) <= 1

    def test_get_test_batches(self):
        test_batches = self.reader.get_test_batches()
        testX = np.zeros((len(self.reader.testX),self.opts.sentence_len))
        testy = np.array(self.reader.testy)
        for ind in range(len(self.reader.testX)):
            exampleX = self.reader.testX[ind][0:self.opts.sentence_len]
            testX[ind,0:len(exampleX)] = exampleX
        
        all_batchX = None
        all_batchy = None
        for batchX, batchy in test_batches:
            assert batchX.shape[0] == self.opts.batch_size
            assert batchy.shape[0] == self.opts.batch_size
            if all_batchX is None:
                all_batchX = batchX
                all_batchy = batchy
            else:
                all_batchX = np.vstack((all_batchX, batchX))
                all_batchy = np.append(all_batchy, batchy)
        all_batchX = all_batchX[0:test_batches.get_num_examples(),:]
        all_batchy = all_batchy[0:test_batches.get_num_examples()]
        assert all_batchX.shape == testX.shape
        assert (all_batchX == testX).all()
        assert all_batchy.shape == testy.shape
        assert (all_batchy == testy).all()

if __name__ == '__main__':
    unittest.main()
