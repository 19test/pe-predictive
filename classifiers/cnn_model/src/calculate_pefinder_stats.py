# Calculate accuracy of PEfinder baseline model using human annotations
import pandas as pd
import numpy as np
import sys

from os.path import join
from classifier import GlobalOpts
from create_partition import get_annot_df

if __name__ == '__main__':
    opts = GlobalOpts('pefinder')

    # Calculate accuracy metrics for baseline pefinder model
    df = get_annot_df(opts)

    # Filter on PE_PRESENT_label
    df = df[df['AllowedAnnotation_name'] == 'PE_PRESENT_label']

    # Filter on positive/negative - exclude uncertain
    df = df[(df['AllowedAnnotation_label'] == 'negative_pe') \
            | (df['AllowedAnnotation_label'] == 'positive_pe')]

    # Change datset format to short form
    df['values'] = df['AllowedAnnotation_label'].apply(
            lambda x : 1 if x == 'positive_pe' else 0)
    df = df.pivot(index='report_id', 
            columns='Annotation_annotator', values='values')
    # Print Annotation Overlap Statistics
    mattlungren_ind = ~pd.isnull(df['mattlungrenMD'])
    pefinder_ind = ~pd.isnull(df['pefinder'])
    num_mattlungren = np.sum(mattlungren_ind)
    num_pefinder = np.sum(pefinder_ind)
    overlap_df = df[mattlungren_ind & pefinder_ind]
    # Uncomment for comparing with just test set
    #overlap_df = overlap_df.reset_index().merge(
    #        pd.read_csv(join(opts.partition_dir, 
    #            'human_annot_only','test.csv')),on='report_id')    
    print 'Num Annotations - MattLungren : %d' % num_mattlungren
    print 'Num Annotations - pefinder : %d' % num_pefinder
    print 'Num Annotations - overlap : %d' % overlap_df.shape[0]

    # Print PEFinder Accuracy Statistics
    result = overlap_df['pefinder'].values
    gt = overlap_df['mattlungrenMD'].values
    test_acc = np.mean(result == gt)
    test_prec = np.mean(gt[result==1])
    test_recall = np.mean(result[gt==1])
    print 'Test Set Evaluation'
    print 'Accuracy : %f' % test_acc
    print 'Precision : %f' % test_prec
    print 'Recall : %f' % test_recall
