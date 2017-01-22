'''
Combines annotations from three reviewers to compare consistency and look
for differences in creation of a gold standard test set
'''
import pandas as pd
from os.path import join
from create_partition import import_task1_data
from ModelFactory import GlobalOpts


if __name__ == '__main__':
    opts = GlobalOpts('gold_standard')
    data = import_task1_data(opts)
    overlap = data[~pd.isnull(data['mattlungrenMD']) 
            & ~pd.isnull(data['mlungrendc76878f480f48f4']) 
            & ~pd.isnull(data['ndm29'])]
   
    diff_inds = ~(overlap['mattlungrenMD']
            + overlap['mlungrendc76878f480f48f4']
            + overlap['ndm29']).isin([0,3])
    diff = overlap[diff_inds]
    num_diff = diff.shape[0]

    report_text = pd.read_csv(opts.report_data_path, sep='\t')
    report_text = report_text[['report_id','rad_report']]
    full_df = diff.merge(report_text, on='report_id')
    assert full_df.shape[0] == num_diff
    print full_df
