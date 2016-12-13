# Check the percentage of words the glove vector covers in the pe set

import pandas as pd
import numpy as np
from collections import defaultdict as dd
from os.path import join, dirname
from reader import preprocess_report


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
    cdata = pd.read_csv(join(classifier_dir, 'chapman-data/chapman_df.tsv'),
            sep="\t",index_col=0)
    sdata = pd.read_csv(join(classifier_dir, 'stanford-data/stanford_df.tsv'),
            sep="\t",index_col=0)
    report_texts = [preprocess_report(report)\
            for report in sdata['rad_report'].values]
    all_words = [word for report in report_texts\
            for word in report.split(' ')]
    all_words = [word.lower() for word in all_words if word != '']
    matched = [word for word in all_words if word in glove_mapping]
    print 'Percent Matched : %f' % (float(len(matched)) / float(len(all_words)))
    # Percent Matched : 0.996035

    print 'original report...'
    print sdata['rad_report'].values[0]
    print 'processed report...'
    print report_texts[0]
    counts = dd(int)
    unmatched = [word for word in all_words if word not in glove_mapping]
    for word in unmatched:
        counts[word] += 1
    df = pd.DataFrame({'word':counts.keys(), 'counts':counts.values()})
    print df.sort_values('counts', ascending=False)
