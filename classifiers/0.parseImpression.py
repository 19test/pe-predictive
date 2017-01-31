import pandas as pd
import numpy as np
import sys
import re

def has_next(all_words, impression_words, index):
    '''
    Return true if next digit in all_words after index is the next sequential
    digit from the last digit present in impression_words
    if impression_words does not have a last digit return False
    if all_words does not have a next digit return False
    '''
    prev_digit = None
    for prev_word in impression_words:
        digit = re.search(r'\d+', prev_word)
        if digit is not None:
            prev_digit = digit.group()
    if prev_digit is None:
        return False
    for next_word in all_words[index::]:
        digit = re.search(r'\d+', next_word)
        if digit is not None:
            if int(prev_digit) + 1 == int(digit.group()):
                return True
            else:
                return False
    return False

def parse_impression(full_report):
    '''
    Return the impression given the full text of the report

    Args:
        full_report : string representing the full report text

    Returns:
        string denoting the impression parsed from the full report.
        all words are converted to lower case
    '''

    percent_upper = lambda string : sum(1 for c in string \
            if c.isupper())/float(len(string))
    impression_words = []
    all_words = full_report.split('-***-')
    start = False
    end = False
    for index in range(len(all_words)):
        word = all_words[index]
        if len(word) == 0:
            continue
        if 'end of impression' in word.lower() \
                or 'summary' in word.lower() \
                or (start and percent_upper(word) < 0.5 \
                and '.' == impression_words[-1][-1] \
                and not has_next(all_words, impression_words, index)):
            end = True
            impression_words.append('END OF IMPRESSION')
            break
        if 'impression:' in word.lower():
            start = True
        if start:
            impression_words.append(word)
    return '\n'.join(impression_words)


if __name__ == '__main__':
    data_path = '../data/stanford_pe.tsv'
    out_path = '../data/stanford_pe_impression.tsv'
    print_examples = False

    data = pd.read_csv(data_path, sep='\t')
    all_reports = data['rad_report'].values.tolist()
    all_impressions = []
    for i in range(len(all_reports)):
        impression = parse_impression(all_reports[i])
        all_impressions.append(impression)
    data['report_text'] = all_impressions
    data = data[['report_id','rad_report','report_text']]
    data.to_csv(out_path, sep='\t')

    # Print example impression parse and full reports
    if print_examples:
        for i in range(len(all_reports)):
            impression = all_impressions[i]
            print '---------------------------'
            if len(impression) == 0 or '1' not in impression:
                print impression
                print 'ORIGINAL REPORT'
                print all_reports[i]
            else:
                print impression
        print len([1 for impression in all_impressions if len(impression) == 0])
        print len([1 for impression in all_impressions if '1' not in impression])
