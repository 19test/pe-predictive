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
        labelX_name = 'rad_report'
        labely_name = 'disease_PEfinder'

        # Import radiology report data
        self.data = pd.DataFrame()
        for path in data_paths:
            data = pd.read_csv(path,sep="\t",index_col=0)
            self.data = self.data.append(data)

        # Preprocess data to remove artifact symbols
        self.data[labelX_name] = self.data[labelX_name].apply(lambda x : preprocess_report(x))

        # Get set of all words across all reports
        self.word_to_id = {}
        self.id_to_word = {}
        word_counter = 0
        for report in self.data[labelX_name].values:
            for word in report.split(' '):
                if word not in self.word_to_id:
                    self.word_to_id[word] = word_counter
                    self.id_to_word[word_counter] = word
                    word_counter += 1

        # Partition data into training, validation, and test set
        test_inds = (self.data['PE_PRESENT_label']=='POSITIVE_PE') \
                | (self.data['PE_PRESENT_label']=='NEGATIVE_PE')
        test_df = self.data[test_inds]
        trainval_df = self.data[np.logical_not(test_inds)] 
        train_size = int(TRAIN_PROPORTION * trainval_df.shape[0])
        train_inds = np.random.choice(range(trainval_df.shape[0]),
                size=train_size, replace=False)
        train_inds = np.array([ind in train_inds for ind in range(trainval_df.shape[0])])
        train_df = trainval_df[train_inds]
        val_df = trainval_df[np.logical_not(train_inds)]

        # tokenize example data
        def tokenize(report):
            return [self.word_to_id[word] for word in report.split(' ')]

        self.trainX = [tokenize(report) for report in train_df[labelX_name]]
        self.trainy = train_df[labely_name].values.tolist()
        self.valX = [tokenize(report) for report in val_df[labelX_name]]
        self.valy = val_df[labely_name].values.tolist()
        self.testX = [tokenize(report) for report in test_df[labelX_name]]
        self.testy = test_df['PE_PRESENT_label'].values.tolist()
        self.testy = [1 if label == 'POSITIVE_PE' else 0 for label in self.testy]
         


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
    
    def get_test_batches(self):
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

        return BatchIterator(self.testX, self.testy,
                self.opts.batch_size, self.opts.sentence_len)
