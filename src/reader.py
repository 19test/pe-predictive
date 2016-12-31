import pandas as pd
import numpy as np
import re


def preprocess_report(report_text):
    '''
    Removes all report text artifacts so that only relevant information remains
    Arguments :
        report_text - String denoting raw report text
    Returns :
        String corresponding to cleaned version of original report
    '''

    # remove reporting artifact
    cleaned_report = report_text.replace("-***-", " ")
    cleaned_report = report_text.replace("*", " ")
    cleaned_report = cleaned_report.lower()
    # separate all words from special characters with whitespace
    lst = re.findall(r"[A-Za-z0-9]+|[^A-Za-z0-9]|\S", cleaned_report)
    cleaned_report = ' '.join(lst)
    # replace multiple spaces with single space
    cleaned_report = re.sub(' +',' ',cleaned_report)
    return cleaned_report

def get_wordvec(vec_path, vocab):
    '''
    Return a dictionary of words -> id and associated embedding matrix
    where words are in vocab.
    Args:
        vec_path - path to glove vector file
        vocab - dictionary where keys are relevant words
    Returns:
        word_to_id : mapping between words and positions in embedding matrix
        embedding_matrix : [id, embedding_len] matrix of word vectors
    '''
    return 


class Reader:

    def __init__(self, data_paths):
        '''
        Creates a reader instance to sample radiology reports
        '''
        TRAIN_PROPORTION = 0.8

        # Import radiology report data
        self.data = pd.DataFrame()
        for path in data_paths:
            data = pd.read_csv(path,sep="\t",index_col=0)
            self.data = self.data.append(data)

        # Preprocess data to remove artifact symbols
        # can also use rad_report here for full report text
        examples = [preprocess_report(report)\
                for report in self.data['rad_report'].values]
        gt_labels = self.data['disease_PEfinder'].values

        # Get set of all words across all reports
        self.word_to_id = {}
        self.id_to_word = {}
        word_counter = 0
        for report in examples:
            for word in report.split(' '):
                if word not in self.word_to_id:
                    self.word_to_id[word] = word_counter
                    self.id_to_word[word_counter] = word
                    word_counter += 1

        # tokenize example data
        tokenized_examples = []
        for report in examples:
            tokenized_examples.append([self.word_to_id[word] for word in report.split(' ')])

        # Partition data into training and validation set
        train_size = int(TRAIN_PROPORTION * len(tokenized_examples))
        train_inds = np.random.choice(range(len(tokenized_examples)),
                size=train_size, replace=False)
        val_inds = [ind for ind in range(len(tokenized_examples))\
                if ind not in train_inds]
        self.train_set = [tokenized_examples[ind] for ind in train_inds]
        self.val_set = [tokenized_examples[ind] for ind in val_inds]
         


    def get_embedding(self, embedding_path):
        '''
        Return np array - embedding matrix of size [vocab_size, embedding_dim]
        using given word vector file with saved weights

        '''
        # Determine word embedding dimension size
        with open(embedding_path, 'r') as f:
            for line in f:
                self.embedding_dim = len(line.split(' ')) - 1
                break

        # fill in word embedding matrix using word vector file
        embedding_contained = {}
        embedding_np = np.zeros((len(self.word_to_id),self.embedding_dim))
        with open(embedding_path, 'r') as f:
            for line in f:
                data = line.split(' ')
                word = data[0]
                if word in self.word_to_id:
                    embedding_np[self.word_to_id[word],:] = \
                            np.array(data[1::]).astype(np.float32)
                    embedding_contained[word] = True

        # fill in all remaining word vectors not found in word vector file
        # with random vectors
        for word_id in range(len(self.word_to_id)):
            word = self.id_to_word[word_id]
            if word not in embedding_contained:
                embedding_np[word_id,:] = np.random.rand(self.embedding_dim)
        return embedding_np


    def sample_train():
        return 0

    def sample_val():
        return 0
