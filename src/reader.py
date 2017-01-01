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

class Reader:

    def __init__(self, opts, data_paths):
        '''
        Creates a reader instance to sample radiology reports
        '''
        self.opts = opts
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
        self.trainX = [tokenized_examples[ind] for ind in train_inds]
        self.trainy = [gt_labels[ind] for ind in train_inds]
        self.valX = [tokenized_examples[ind] for ind in val_inds]
        self.valy = [gt_labels[ind] for ind in val_inds]
         


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


    def _sample(self, set_name, balance=True):
        assert set_name in ['train', 'val']
        setX = self.trainX if set_name == 'train' else self.valX
        sety = self.trainy if set_name == 'train' else self.valy
        if balance:
            assert self.opts.batch_size % 2 == 0
            pos_inds = np.nonzero(np.array(sety))[0]
            neg_inds = np.nonzero(1 - np.array(sety))[0]
            sel_pos = np.random.choice(pos_inds,
                    size=self.opts.batch_size/2, replace=True)
            sel_neg = np.random.choice(neg_inds,
                    size=self.opts.batch_size/2, replace=True)
            inds = np.append(sel_pos, sel_neg)
            np.random.shuffle(inds)
        else:
            inds = np.random.choice(range(len(setX)),
                    size=self.opts.batch_size, replace=False)
        batchX = np.zeros((self.opts.batch_size, self.opts.sentence_len))
        batchy = np.zeros(self.opts.batch_size) 
        for i in range(self.opts.batch_size):
            ind = inds[i]
            exampleX = setX[ind]
            if len(exampleX) > self.opts.sentence_len:
                exampleX = exampleX[0:self.opts.sentence_len]
            batchX[i,0:len(exampleX)] = np.array(exampleX)
            batchy[i] = sety[ind] 
        return batchX, batchy

    def sample_train(self):
        return self._sample('train')

    def sample_val(self):
        return self._sample('val')
