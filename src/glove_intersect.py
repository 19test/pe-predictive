# Check the percentage of words the glove vector covers in the pe set

import pandas as pd
import numpy as np
from os.path import join, dirname



def get_glove_words(glove_path):
    words = []
    with open(glove_path, 'r') as f:
        for line in f:
            words.append(line.strip().split(' ')[0])
    return words

if __name__ == '__main__':
    project_dir = join(dirname(__file__), '..')
    classifier_dir = join(project_dir, 'classifiers')
    glove_path = join(project_dir, 'data', 'glove.42B.300d.txt')

    print 'Importing glove words...'
    glove_words = get_glove_words(glove_path)
    glove_mapping = {word : True for word in glove_words}
    
    print 'Calculating match...'
    # Read in chapman and stanford data
    cdata = pd.read_csv(join(classifier_dir, 'chapman-data/chapman_df.tsv'),sep="\t",index_col=0)
    sdata = pd.read_csv(join(classifier_dir, 'stanford-data/stanford_df.tsv'),sep="\t",index_col=0)
    all_words = [word for report in sdata['rad_report'].values for word in report.split(' ')]
    all_words = [word.lower() for word in all_words if word != '']
    matched = [word for word in all_words if word in glove_mapping]
    print 'Percent Matched : %f' % (float(len(matched)) / float(len(all_words)))
    # Percent Matched : 0.761822
