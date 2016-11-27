import pandas as pd
import numpy as np


def preprocess_report(report_text):
    '''
    Removes all report text artifacts so that only relevant information remains
    Arguments :
        report_text - String denoting raw report text
    Returns :
        String corresponding to cleaned version of original report
    '''
    cleaned_report = report_text.replace("_***_", ".")
    cleaned_report = cleaned_report.upper()
    #cleaned_report = cleaned_report.replace("\n","")
    return cleaned_report

class Sampler:

    def __init__(self):
        self.data = pd.DataFrame()

    def add_data(self, path): 
        data = pd.read_csv(path,sep="\t",index_col=0)
        self.data = self.data.append(data)

    def add_embeddings(self, path):
        # process relevant embeddings into a hash map

    def get_embedding_len(self):
        # Get word vector representation length
        return 0

    def split_train(self):
        # Split data into train and validation sets

    def initialize(self):
        # Preprocess data to remove artifact symbols
        # can also use rad_report here for full report text
        examples = [preprocess_report(report) for report in all_data['impression'].values]
        gt_labels = all_data['disease_PEfinder'].values

    def sample_train():
        return 0

    def sample_val():
        return 0
