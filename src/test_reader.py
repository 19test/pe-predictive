import unittest
from os.path import join, dirname
from reader import Reader

class TestReader(unittest.TestCase):
    def setUp(self): 
        # Directory structure
        self.project_dir = join(dirname(__file__), '..')
        self.classifier_dir = join(self.project_dir, 'classifiers')
        self.checkpoint_dir = join(self.project_dir, 'checkpoints')
        self.reader = Reader()
        self.reader.add_data()

    def test_word_embeddings(self):
        
        assert False

if __name__ == '__main__':
    unittest.main()
