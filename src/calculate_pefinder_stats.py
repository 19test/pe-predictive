# Calculate accuracy of PEfinder baseline model using human annotations
import pandas as pd
import numpy as np

from os.path import join
from classifier import GlobalOpts

if __name__ == '__main__':
        opts = GlobalOpts('pefinder')

        # Calculate accuracy metrics for baseline pefinder model
        data = pd.read_csv(
                join(opts.project_dir, 'data', 'stanford_pe.tsv'),
                sep='\t')
        test_inds = (data['PE_PRESENT_label']=='POSITIVE_PE') \
                | (data['PE_PRESENT_label']=='NEGATIVE_PE')
        test_df = data[test_inds]
        result = test_df['disease_PEfinder']
        gt = np.array([1 if label == 'POSITIVE_PE' else 0 \
                for label in test_df['PE_PRESENT_label']])

        test_acc = np.mean(result == gt)
        test_prec = np.mean(gt[result==1])
        test_recall = np.mean(result[gt==1])
        print 'Test Set Evaluation'
        print 'Accuracy : %f' % test_acc
        print 'Precision : %f' % test_prec
        print 'Recall : %f' % test_recall
