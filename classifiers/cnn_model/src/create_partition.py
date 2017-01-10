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

    data = pd.read_csv(opts.data_path, sep='\t')

    if args.partition == 'pefinder_augment': 
        # Partition data into training, validation, and test set
        test_inds = (data['PE_PRESENT_label']=='POSITIVE_PE') \
                | (data['PE_PRESENT_label']=='NEGATIVE_PE')
        test_df = data[test_inds]
        test_df['label'] = test_df['PE_PRESENT_label'].apply(lambda x :
                1 if x == 'POSITIVE_PE' else 0)
        #trainval_df = data[np.logical_not(test_inds)] 
        # TODO : WARNING - current overlap of trainval and test data
        trainval_df = data
        trainval_df['label'] = trainval_df['disease_PEfinder']
        train_size = int(TRAIN_PROPORTION * trainval_df.shape[0])
        train_inds = np.random.choice(range(trainval_df.shape[0]),
                size=train_size, replace=False)
        train_inds = np.array([ind in train_inds \
                for ind in range(trainval_df.shape[0])])
        train_df = trainval_df[train_inds]
        val_df = trainval_df[np.logical_not(train_inds)]

    elif args.partition == 'human_annot_only':
        
        manual_inds = (data['PE_PRESENT_label']=='POSITIVE_PE') \
                | (data['PE_PRESENT_label']=='NEGATIVE_PE')
        manual_annot = data[manual_inds].reset_index()
        manual_annot['label'] = manual_annot['PE_PRESENT_label'].apply(
                lambda x : 1 if x == 'POSITIVE_PE' else 0)
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
