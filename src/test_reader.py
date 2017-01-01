import unittest
import numpy as np
from os.path import join, dirname
from reader import Reader

from classifier import GlobalOpts


class TestReader(unittest.TestCase):
    def setUp(self):
        self.opts = GlobalOpts('test')
        data_paths = [join(self.opts.classifier_dir, 'stanford-data/stanford_df.tsv')]
        self.reader = Reader(opts=self.opts, data_paths=data_paths)

    def test_split_train(self):
        train_size = len(self.reader.train_set)
        val_size = len(self.reader.val_set)
        assert self.reader.data.shape[0] == train_size + val_size        

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
        result = self.reader.sample_train()
        assert result.shape == (self.opts.batch_size, self.opts.sentence_len)

    def test_sample_val(self):
        result = self.reader.sample_val()
        assert result.shape == (self.opts.batch_size, self.opts.sentence_len)

if __name__ == '__main__':
    unittest.main()
