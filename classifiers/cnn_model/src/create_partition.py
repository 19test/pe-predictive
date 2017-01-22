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
from ModelFactory import GlobalOpts

# proportion of total dataset
TEST_PROPORTION = 0.2
# proportion of train/val dataset without test
TRAIN_PROPORTION = 0.8

def save_partition(df, label_lst, filepath):
    relevant_columns = ['report_id'] + label_lst
    df = df[relevant_columns]
    print 'Saved %d cols to : %s'%(df.shape[0], filepath)
    df.to_csv(filepath)

def import_task1_data(opts):
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
            '8_mlungrendc76878f480f48f4_annotations.tsv',
            '8_ndm29_annotations.tsv',
            '8_pefinder_annotations.tsv',
            '9_mattlungrenMD_annotations.tsv',
            '9_mlungrendc76878f480f48f4_annotations.tsv',
            '9_ndm29_annotations.tsv',
            '9_pefinder_annotations.tsv']
    filepath_lst = [join(opts.task1_annot_dir, filepath) \
            for filepath in filepath_lst]
    data = pd.DataFrame()
    for filepath in filepath_lst:
        curr = pd.read_csv(filepath, sep='\t')
        data = data.append(curr)
    data = data.drop(data.columns[0],axis=1)
    data = data.drop_duplicates()

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
    return data

def import_task2_data(opts):
    filepath_lst = ['16_mlungrendc76878f480f48f4_annotations.tsv']
    filepath_lst = [join(opts.task2_annot_dir, filepath) \
            for filepath in filepath_lst]
    data = pd.DataFrame()
    for filepath in filepath_lst:
        curr = pd.read_csv(filepath, sep='\t')
        data = data.append(curr)
    data = data.drop(data.columns[0],axis=1)
    data = data.drop_duplicates()
    data = data.pivot(index='report_id', 
            columns='AllowedAnnotation_name', values='AllowedAnnotation_label')
    data = data.reset_index()
    data = data[data['LOOKING_FOR_PE_label'].isin(['pe_study','nonpestudy'])]
    assert data.shape[0] == 999
    labelA_mapping = {
            'nonpestudy' : 0,
            'pe_study' : 1
            }
    labelB_mapping = {
            None : 0,
            'CENTRAL' : 1,
            'MASSIVE_SADDLE' : 2,
            'segmental' : 3,
            ' subsegmental_only' : 4
            }
    data['LOOKING_FOR_PE_label'] = data['LOOKING_FOR_PE_label'].apply(
            lambda x : labelA_mapping[x])
    data['PE_BURDEN_label'] = data['PE_BURDEN_label'].apply(
            lambda x : labelB_mapping[x])
    return data

def random_split(df, proportion):
    '''
    Randomly splits the rows of df into two separate dataframes with given proportion
    Args:
        df - input pandas data frame to split
        proportion - [0,1] float representing proportion size
                        of first returned data frame
    Returns:
        Two pandas dataframe where the size of the first is proportion
    '''
    df = df.reset_index(drop=True)
    sizeA = int(proportion * df.shape[0])
    sizeB = df.shape[0] - sizeA
    inds = range(df.shape[0])
    indsA = np.random.choice(inds, size=sizeA, replace=False)
    indsB = np.array([ind for ind in inds if ind not in indsA])
    dfA = df.loc[indsA].reset_index(drop=True)
    dfB = df.loc[indsB].reset_index(drop=True)
    return dfA, dfB

if __name__ == '__main__':
    opts = GlobalOpts('create_partitions')
    parser = argparse.ArgumentParser(
            description='Create and save data partition indicies')
    parser.add_argument('--partition',
            help='Way to split data into train/val/test set', required=True)
    args = parser.parse_args()

    out_dir = join(opts.partition_dir, args.partition)


    if args.partition == 'pefinder_augment':
        '''
        Test set - all annotations completed by mattlungrenMD with
                    label being his annotations
        Train/val set - remaining set with label being output of pefinder
        '''

        data = import_task1_data(opts)
        test_df = data[~pd.isnull(data['mattlungrenMD'])]
        test_df['label'] = test_df['mattlungrenMD']
        label_lst = ['label']
        trainval_df = data[pd.isnull(data['mattlungrenMD'])]
        
        trainval_df['label'] = trainval_df['pefinder']
        train_df, val_df = random_split(trainval_df,
                proportion=TRAIN_PROPORTION)

    elif args.partition == 'task1_human':
        '''
        Only contains set annotated by mattlungrenMD
        test set - random TEST_PROPORTION of test set
        train/val set - random split of remainder using TRAIN_PROPORTION
        '''
        data = import_task1_data(opts)
        manual_annot = data[~pd.isnull(data['mattlungrenMD'])].reset_index()
        manual_annot['label'] = manual_annot['mattlungrenMD']
        label_lst = ['label']
        total_size = manual_annot.shape[0]

        train_df, val_df = random_split(manual_annot,
                proportion=TRAIN_PROPORTION)
        assert train_df.shape[0] + val_df.shape[0] \
                == total_size
    elif args.partition == 'task2_human':
        data = import_task2_data(opts)
        data['label_pe'] = data['LOOKING_FOR_PE_label']
        data['label_burden'] = data['PE_BURDEN_label']
        label_lst = ['label_pe', 'label_burden']
        train_df, val_df = random_split(data,
                proportion=TRAIN_PROPORTION)
        assert train_df.shape[0] + val_df.shape[0] \
                == data.shape[0]
    else:
        raise Exception('Split type unsupported : %s' % args.partition)

    if exists(out_dir):
        raise Exception('Target directory already exists : %s' % out_dir)
    else:
        os.makedirs(out_dir)

    save_partition(train_df, label_lst, join(out_dir, 'train.csv'))
    save_partition(val_df, label_lst, join(out_dir, 'val.csv')) 
