'''
Combines annotations from three reviewers to compare consistency and look
for differences in creation of a gold standard test set
'''
import pandas as pd

from create_partition import get_annotated_pe
from classifier import GlobalOpts


if __name__ == '__main__':
    opts = GlobalOpts('gold_standard')
    data = get_annotated_pe(opts)
    overlap = data[~pd.isnull(data['mattlungrenMD']) 
            & ~pd.isnull(data['mlungrendc76878f480f48f4']) 
            & ~pd.isnull(data['ndm29'])]
   
    diff_inds = ~(overlap['mattlungrenMD']
            + overlap['mlungrendc76878f480f48f4']
            + overlap['ndm29']).isin([0,3])
    diff = overlap[diff_inds]
    print diff
    print diff.shape
