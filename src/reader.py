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

    def __init__(self, data_paths, embedding_path):
        '''
        Creates a reader instance to sample radiology reports
        '''

        # Import radiology report data
        self.data = pd.DataFrame()
        for path in data_paths:
            data = pd.read_csv(path,sep="\t",index_col=0)
            self.data = self.data.append(data)

        # Get set of all words across all reports
        vocab = {}
        for report in self.data['rad_report']:
            processed = process_report(report)
            for word in processed.split(' '):
                vocab[word] = True
        self.word_to_id, self.embedding_matrix = get_wordvec(embedding_path, vocab)

    def get_embedding_len(self):
        # Get word vector representation length
        return 0

    def split_train(self):
        # Split data into train and validation sets
        return 0

    def initialize(self):
        # Preprocess data to remove artifact symbols
        # can also use rad_report here for full report text
        examples = [preprocess_report(report)\
                for report in all_data['impression'].values]
        gt_labels = all_data['disease_PEfinder'].values

    def sample_train():
        return 0

    def sample_val():
        return 0

if __name__ == '__main__':
    reader = Reader()
