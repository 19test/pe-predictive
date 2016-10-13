#!/bin/python
# Note, this should be run within the Docker image provided to have all dependendencies, see
# ../Dockerfile in the base repo along with the README.md for setup instructions
# Here we are going to try using sklean Countvectorizer


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn import ensemble

import pandas
import pickle
import numpy
import os

# Let's make sure we are working from CODE HOME
CODE_HOME = os.environ["CODE_HOME"]
os.chdir(CODE_HOME)

# Read in chapman and stanford data
cdata = pandas.read_csv('chapman-data/chapman_df.tsv',sep="\t",index_col=0)
sdata = pandas.read_csv('stanford-data/stanford_df.tsv',sep="\t",index_col=0)

# Batches 1 and 2 are not independent, we will only use larger of the two
batches = [2.,3.,4.]
sdata = sdata[sdata.batch.isin(batches)]
# 691

def predictPE(reports,y,cv=10,labels=None):
    '''predictPE will vectorize text and then run N fold cross validation 
    with an ensemble tree classifier to predict PE
    :param reports: should be a list or series of report text to be vectorized
    :param y: should be the actual rx labels (right now in Neg and Pos)
    :param labels: should be the unique labels (just for column/row names of confusion)
    '''
    if labels == None:
        labels = ['Neg','Pos']
    countvec = CountVectorizer(ngram_range=(1,2),stop_words="english")
    X = countvec.fit_transform(reports)
    clf = ensemble.ExtraTreesClassifier()
    scores = cross_val_score(clf, X, y, cv=cv)
    predictions = cross_val_predict(clf, X, y, cv=cv)
    confusion = makeConfusion(y_true=y,
                              y_pred=predictions,
                              labels=labels)
    result = {"scores":scores,
              "cv":cv,
              "confusion":confusion}
    return result

def print_result(result):
    for rkey,res in result.items():
        print("%s:" %rkey)
        print(res)
        print('\n')

def replace_text(textList,findList,replacewith):
    if isinstance(findList,str):
        return [x.replace(findList,replacewith) for x in textList]
    else:
        for find in findList:
            textList = [x.replace(find,replacewith) for x in textList]
        return textList


# Predict PE for Stanford Data (IMPRESSIONS):
reports = sdata['impression']
y = sdata['disease_state']
stanford_result = predictPE(reports=reports,y=y)
print_result(stanford_result)

# scores:
# [ 0.91428571  0.88571429  0.9         0.91428571  0.89855072  0.89855072
#   0.91304348  0.91176471  0.91176471  0.89705882]


# confusion:
#     Neg  Pos
# Neg  612    2
# Pos   66   11

# cv:
# 10

# Predict PE for Stanford Data (FULL REPORTS):
reports = sdata['rad_report']
y = sdata['disease_state']
stanford_result = predictPE(reports=reports,y=y)
print_result(stanford_result)

# scores:
#[ 0.91428571  0.88571429  0.9         0.91428571  0.88405797  0.91304348
#  0.88405797  0.92647059  0.89705882  0.92647059]


# confusion:
#    Neg  Pos
# Neg  614    0
# Pos   74    3


# cv:
# 10


# Predict PE for Chapman data:
cdata = cdata[cdata.disease_state.isnull()==False]
reports = cdata['impression'].tolist()

# Clean up text
getRidOf = ['[Report de-identified (Limited dataset compliant) by De-ID v.6.21.01.0]',
            'NULL']
reports = replace_text(reports,getRidOf,'')

y = cdata['disease_state']
chapman_result = predictPE(reports=reports,y=y)
print_result(chapman_result)

# scores:
# [ 0.86206897  0.85057471  0.89655172  0.8255814   0.84883721  0.8372093
#   0.88235294  0.85882353  0.87058824  0.92941176]


# confusion:
#      Neg  Pos
# Neg  538   28
# Pos   91  202


# cv:
# 10


