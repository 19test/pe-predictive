import pandas as pd
import numpy as np
import re

from os.path import join

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

def split_data(data, partition_dir, partition):
    train_partition = pd.read_csv(join(partition_dir, partition, 'train.csv'))
    val_partition = pd.read_csv(join(partition_dir, partition, 'val.csv'))
    test_partition = pd.read_csv(join(partition_dir, partition, 'test.csv'))
    train_df = data.merge(train_partition, on='report_id')
    val_df = data.merge(val_partition, on='report_id')
    test_df = data.merge(test_partition, on='report_id')
    return train_df, val_df, test_df


class Reader:
    '''
    Input class that splits dataset into train/val/test and tokenizes
    the words into embedded vectors. Also handles importing of Glove vectors.
    '''

    def __init__(self, opts):
        '''
        Creates a reader instance to sample radiology reports
        '''
        self.opts = opts
        labelX_name = 'rad_report' if opts.full_report else 'report_text'
        labely_name = 'label'

        # Import radiology report data
        report_data = pd.read_csv(opts.report_data_path,sep="\t",index_col=0)

        # Preprocess report_data to remove artifact symbols
        report_data[labelX_name] = report_data[labelX_name].apply(
                lambda x : preprocess_report(x))
        # Get set of all words across all reports
        self.word_to_id = {}
        self.id_to_word = {}
        word_counter = 0
        for report in report_data[labelX_name].values:
            for word in report.split(' '):
                if word not in self.word_to_id:
                    self.word_to_id[word] = word_counter
                    self.id_to_word[word_counter] = word
                    word_counter += 1

        train_df, val_df, test_df = split_data(report_data,
                opts.partition_dir, opts.partition)


        self.trainX = [self._tokenize(report) for report in train_df[labelX_name]]
        self.trainy = train_df[labely_name].values.tolist()
        self.valX = [self._tokenize(report) for report in val_df[labelX_name]]
        self.valy = val_df[labely_name].values.tolist()
        self.testX = [self._tokenize(report) for report in test_df[labelX_name]]
        self.testy = test_df[labely_name].values.tolist()
        
        # Print stats on split
        print 'Train - Size : %d - Pct_POS : %f' % \
                (len(self.trainy), np.mean(self.trainy))
        print 'Val - Size : %d - Pct_POS : %f' % \
                (len(self.valy), np.mean(self.valy))
        print 'Test - Size : %d - Pct_POS : %f' % \
                (len(self.testy), np.mean(self.testy))

    # tokenize example data
    def _tokenize(self, report):
        return [self.word_to_id[word] for word in report.split(' ')]

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
   
    def get_raw_batches(self, report_lst):
        # convert report_lst to list of word id sequences
        reports = [preprocess_report(report) for report in report_lst]
        reports = [self._tokenize(report) for report in reports]
        # return batch of sequences
        return BatchIterator(reports, np.random.randint(2,size=len(reports)),
                self.opts.batch_size, self.opts.sentence_len)

    def get_test_batches(self):
        return BatchIterator(self.testX, self.testy,
                self.opts.batch_size, self.opts.sentence_len)

class BatchIterator(object):
    def __init__(self, setX, sety, batch_size, sentence_len):
        self.setX = setX
        self.sety = sety
        self.batch_size = batch_size
        self.sentence_len = sentence_len
        self.i = 0
    def __iter__(self):
        return self
    def get_num_examples(self):
        return len(self.setX)
    def next(self):
        if self.i >= self.get_num_examples():
            raise StopIteration()
        batchX = np.zeros((self.batch_size, self.sentence_len))
        batchy = np.zeros(self.batch_size)
        for ind in range(self.batch_size):
            exampleX = self.setX[self.i][0:self.sentence_len]
            batchX[ind,0:len(exampleX)] = exampleX
            batchy[ind] = self.sety[self.i]
            self.i += 1
            if self.i >= self.get_num_examples():
                break
        return batchX, batchy
