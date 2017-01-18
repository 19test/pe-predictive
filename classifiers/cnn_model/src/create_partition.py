'''
Saves file which stores a random partition of a given type to the partition folder

The purpose of this process is to hold the train/val/test sets constant
when running different models across multiple machines
'''

import pandas as pd
import numpy as np
import argparse
import os

from os.path import join, exists
from classifier import GlobalOpts

# proportion of total dataset
TEST_PROPORTION = 0.2
# proportion of train/val dataset without test
TRAIN_PROPORTION = 0.8

def save_partition(df, filepath):
    df = df[['report_id','label']]
    df.to_csv(filepath)

def get_annot_df(opts):
    '''
    Returns annotation df in long form with the following columns
        report_id - unique id for report
        AllowedAnnotation_name : Type of annotation
            PE_PRESENT_label, ACUITY, etc...
        AllowedAnnotation_label : Value of annotation
        Annotation_annotator : name of annotator
    Args :
        opts - global options objects which contains file path structure
    '''
    # Aggregate all annotation data
    filepath_lst = ['8_mattlungrenMD_annotations.tsv',
            '8_pefinder_annotations.tsv',
            '9_mattlungrenMD_annotations.tsv',
            '9_pefinder_annotations.tsv']
    filepath_lst = [join(opts.data_dir, filepath) for filepath in filepath_lst]
    data = pd.DataFrame()
    for filepath in filepath_lst:
        curr = pd.read_csv(filepath, sep='\t')
        data = data.append(curr)
    data = data.drop(data.columns[0],axis=1)
    data = data.drop_duplicates()
    return data

if __name__ == '__main__':
    opts = GlobalOpts('create_partitions')
    parser = argparse.ArgumentParser(
            description='Create and save data partition indicies')
    parser.add_argument('--partition',
            help='Way to split data into train/val/test set', required=True)
    args = parser.parse_args()

    out_dir = join(opts.partition_dir, args.partition)
    if exists(out_dir):
        raise Exception('Target directory already exists : %s' % out_dir)
    else:
        os.makedirs(out_dir)

    data = get_annot_df(opts)

    # Filter on PE_PRESENT_label
    data = data[data['AllowedAnnotation_name'] == 'PE_PRESENT_label']

    # Filter on positive/negative - exclude uncertain
    data = data[(data['AllowedAnnotation_label'] == 'negative_pe') \
            | (data['AllowedAnnotation_label'] == 'positive_pe')]

    # Change datset format to short form
    data['values'] = data['AllowedAnnotation_label'].apply(
            lambda x : 1 if x == 'positive_pe' else 0)
    data = data.pivot(index='report_id', 
            columns='Annotation_annotator', values='values')
    data = data.reset_index()

    if args.partition == 'pefinder_augment':
        '''
        Test set - all annotations completed by mattlungrenMD with
                    label being his annotations
        Train/val set - remaining set with label being output of pefinder
        '''
        test_df = data[~pd.isnull(data['mattlungrenMD'])]
        test_df['label'] = test_df['mattlungrenMD']
        trainval_df = data[pd.isnull(data['mattlungrenMD'])]
        
        trainval_df['label'] = trainval_df['pefinder']
        train_size = int(TRAIN_PROPORTION * trainval_df.shape[0])
        train_inds = np.random.choice(range(trainval_df.shape[0]),
                size=train_size, replace=False)
        train_inds = np.array([ind in train_inds \
                for ind in range(trainval_df.shape[0])])
        train_df = trainval_df[train_inds]
        val_df = trainval_df[~train_inds]

    elif args.partition == 'human_annot_only':
        '''
        Only contains set annotated by mattlungrenMD
        test set - random TEST_PROPORTION of test set
        train/val set - random split of remainder using TRAIN_PROPORTION
        '''
        manual_annot = data[~pd.isnull(data['mattlungrenMD'])].reset_index()
        manual_annot['label'] = manual_annot['mattlungrenMD']
        total_size = manual_annot.shape[0]
        test_size = int(TEST_PROPORTION * total_size)
        train_size = int(TRAIN_PROPORTION * (total_size - test_size))
        val_size = total_size - test_size - train_size
        inds = range(total_size)
        np.random.shuffle(inds)
        train_inds = inds[0:train_size]
        val_inds = inds[train_size:train_size+val_size]
        test_inds = inds[train_size+val_size::]
        train_df = manual_annot.loc[train_inds,:]
        val_df = manual_annot.loc[val_inds,:]
        test_df = manual_annot.loc[test_inds,:]
        assert len(train_inds) + len(val_inds) + len(test_inds) == total_size    
    else:
        raise Exception('Split type unsupported : %s' % args.partition)
    save_partition(train_df, join(out_dir, 'train.csv'))
    save_partition(val_df, join(out_dir, 'val.csv')) 
    save_partition(test_df, join(out_dir, 'test.csv'))
