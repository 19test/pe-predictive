import tensorflow as tf
import pandas as pd
import numpy as np
import argparse

from os.path import join, dirname
from reader import Reader
from classifier import ModelFactory
from create_partition import get_annot_pe


if __name__ == '__main__':
    tf.set_random_seed(1)
    np.random.seed(1)

    parser = argparse.ArgumentParser(
        description='Text Classification Models for PE-Predictive Project')
    parser.add_argument('--arch', help='Network architecture',
            type=str, required=True)
    parser.add_argument('--name',
            help='Name of directory to place output files in',
            type=str, required=True)
    # unused
    parser.add_argument('--partition',
            help='Way to split data into train/val/test set',
            default='human_annot_only')
    parser.add_argument('--input_path',
            help='input csv file with reports to process',
            type=str, required=True)
    parser.add_argument('--output_path',
            help='Path of output csv with additional pred column from input',
            type=str, required=True)
    parser.add_argument('-full_report', action='store_true',
            help='use full report text - otherwise use impression input')
    args = parser.parse_args()
    factory = ModelFactory(args.arch, args.name)
    opts = factory.get_opts(args)

    reader = Reader(opts=opts)
    embedding_np = reader.get_embedding(opts.glove_path)
    model = factory.get_model(embedding_np)
    model.restore_weights()

    sep = '\t' if args.input_path.split('.')[-1]=='tsv' else ','
    input_reports = pd.read_csv(args.input_path, sep=sep)

    labelX_name = 'rad_report' if opts.full_report else 'report_text'
    inputX = input_reports[labelX_name]
    raw_batch_iter = reader.get_raw_batches(inputX)
    total_size = raw_batch_iter.get_num_examples() 

    result = None
    for batchX, _ in raw_batch_iter:
        output = model.predict(batchX)
        if result is None:
            result = output
        else:
            result = np.append(result, output)
    result = result[0:total_size]
    assert len(result) == total_size
    assert len(result) == input_reports.shape[0]

    input_reports['pred'] = result

    annotations = get_annotated_pe(opts)
    output_results = input_reports.merge(annotations, on='report_id')
    output_results = output_results[['report_id',
        'pefinder','mattlungrenMD','pred']]

    output_results.to_csv(args.output_path)
