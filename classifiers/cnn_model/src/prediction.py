'''
Wrapper around classifier class which exposes a function to run the models
on a given input from a python function
'''
import pandas as pd
import numpy as np
import tensorflow as tf
import argparse
from reader import Reader
from ModelFactory import ModelFactory


class default_args:
    def __init__(self):
        self.arch = 'cnn_word'
        self.name = 'cnn_word_task1_human'
        self.task_num = 1
        self.error_analysis = False

def predict(input_list, args=None):
    tf.reset_default_graph()
    if args is None:
        args = default_args()

    factory = ModelFactory(args.arch, args.name)
    opts = factory.get_opts(args)

    reader = Reader(opts=opts, inputX=input_list)
    embedding_np = reader.get_embedding(opts.glove_path)
    model = factory.get_model(embedding_np, task_num=opts.task_num)

    input_batches = reader.get_raw_batches(input_list)

    model.restore_weights()
    total_size = input_batches.get_num_examples() 
    result = None
    for batchX, _ in input_batches:
        output = model.predict(batchX)
        if result is None:
            result = output
        else:
            result = np.vstack((result, output))
    result = result[0:total_size,:]
    assert result.shape[0] == total_size
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Text Classification Models for PE-Predictive Project')
    parser.add_argument('--arch', help='Network architecture',
            type=str, required=True)
    parser.add_argument('--name',
            help='Name of directory to place output files in',
            type=str, required=True)
    parser.add_argument('--task_num',
            help='Either task 1 - impressions, or task 2 - full report text',
            type=int, required=True)
    # used for predict runtype
    parser.add_argument('--input_path',
            help='input csv file with reports to process', type=str)
    parser.add_argument('--output_path',
            help='Path of output csv with additional pred column from input',
            type=str)
    args = parser.parse_args()

    sep = '\t' if args.input_path.split('.')[-1]=='tsv' else ','
    input_reports = pd.read_csv(args.input_path, sep=sep)

    labelX_name = 'rad_report' if args.task_num==2 else 'report_text'
    inputX = input_reports[labelX_name]
    predictions = predict(inputX, args)
    for i in range(predictions.shape[1]):
        input_reports['pred_%d'%i] = predictions[:,i]
    input_reports.to_csv(args.output_path)
