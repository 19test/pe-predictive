#!/bin/python
# Note, this should be run within the Docker image provided to have all dependendencies, see
# ../Dockerfile in the base repo along with the README.md for setup instructions
# Here we are going to try using sklean Countvectorizer


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score, \
    cross_val_predict, train_test_set, Kfold
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

def getImportance(clf,lookup,save_top=None):
    '''getImporance will get the important features based on a lookup 
    :param clf: the fit classifier to assess
    :param lookup: the lookup (keys are indices, values vocab words) to use
    :param save_top: how many of top N to save
    '''
    importance_df = pandas.DataFrame()
    importance_df['importances'] = clf.feature_importances_
    importance_df['std'] = numpy.std([clf.feature_importances_ for tree in clf.estimators_],axis=0)
    indices = numpy.argsort(importance_df["importances"])[::-1]
    importance_df['vocabulary'] = [lookup[x] for x in indices]
    importance_df.sort(columns=['importances'],inplace=True,ascending=False)
    if save_top != None:
        importance_df = importance_df.loc[importance_df.index[0:save_top]]
    return importance_df


def predictPE(reports,y,cv=10,labels=None):
    '''predictPE will vectorize text and then run N fold cross validation 
    with an ensemble tree classifier to predict PE
    :param reports: should be a list or series of report text to be vectorized
    :param y: should be the actual rx labels (right now in Neg and Pos)
    :param labels: should be the unique labels (just for column/row names of confusion)
    '''
    if labels == None:
        labels = ['Neg','Pos']

    # Generate vectors of counts
    countvec = CountVectorizer(ngram_range=(1,2),
                               stop_words="english")

    # Fit the model
    X = countvec.fit_transform(reports)

    # Convert lookup of terms to indices, to indices to terms
    lookup = {val:key for key,val in countvec.vocabulary_.items()}

    splits = KFold(n_splits=cv,shuffle=True)
    for train_idx, test_idx in splits.split(X):
        trainX = X[train_idx]
        # index the labels with test_idx
        y.index=range(0,len(y))
        trainy = y.loc[train_idx].tolist()     
        
        # 1. Fit the classifier
        clf = ensemble.ExtraTreesClassifier(class_weight='balanced')
        clf.fit(X,y)
        # ExtraTreesClassifier(bootstrap=False, class_weight='balanced',
        #   criterion='gini', max_depth=None, max_features='auto',
        #   max_leaf_nodes=None, min_impurity_split=1e-07,
        #   min_samples_leaf=1, min_samples_split=2,
        #   min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
        #   oob_score=False, random_state=None, verbose=0, warm_start=False)

        # 2. Which features are important?
        # NOTE: we have a lot of numbers in here - might want to use regex to filter
        importances = getImportance(clf=clf,
                                    lookup=lookup,
                                    save_top=100)

        # 3. How well did we do? Test on test
        testX = X[test_idx]
        testy = y.loc[test_idx].tolist()        
        predictions = clf.predict(testX).tolist()
        labels = numpy.unique(testy + predictions).tolist()
        confusion = makeConfusion(y_true=testy,
                                  y_pred=predictions,
                                  labels=labels)

        # 4. Which ones did we get wrong?

        # For each, let's save scores, cv, confusion, wrong
        result = {"scores":scores,
                  "cv":cv,
                  "confusion":confusion,
                  "importances":importances}

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


###############################################################
# PE PREDICTION [Neg,Pos]
###############################################################


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


###############################################################
# PE PREDICTION [oher labels minus probably positive]
###############################################################

reports = sdata['impression']
y = sdata['disease_state_label'].copy()

# We only have N=2 for "probably positive" so we will change to 
# "definitely positive" to allow for more cross validation groups
# This would only hurt us in making it harder to predict PE if
# the signal in the "probably positive" isn't as strong
y[y=="probably positive"] = "definitely positive"
labels = y.unique().tolist()

stanford_result = predictPE(reports=reports,y=y,labels=labels)
print_result(stanford_result)

# scores:
#[ 0.88732394  0.87142857  0.85714286  0.85507246  0.85507246  0.88405797
#  0.85507246  0.91176471  0.86764706  0.88235294]


# confusion:
#                     definitely negative  definitely positive  \
# definitely negative                    0                   13   
# definitely positive                    0                  587   
# probably negative                      0                   61   
# Indeterminate                          0                   10   

#                      probably negative  Indeterminate  
# definitely negative                  0              0  
# definitely positive                  4              0  
# probably negative                   16              0  
# Indeterminate                        0              0  

# cv:
# 10


reports = sdata['rad_report']
stanford_result = predictPE(reports=reports,y=y,labels=labels)
print_result(stanford_result)

# {'scores': array([ 0.84507042,  0.84285714,  0.84285714,  0.86956522,  0.85507246,
   # 0.88405797,  0.89855072,  0.89705882,  0.86764706,  0.88235294]), 'confusion':                      definitely negative  definitely positive  \
# definitely negative                    0                   12   
# definitely positive                    0                  591   
# probably negative                      0                   73   
# Indeterminate                          0                   10   

#                      probably negative  Indeterminate  
# definitely negative                  1              0  
# definitely positive                  0              0  
# probably negative                    4              0  
# Indeterminate                        0              0  , 'cv': 10}

# This gives us more fine grainted predicted, of course at a loss of "performance"



###############################################################
# QUALITY PREDICTION [Diagnostic=589,Limited=97]
###############################################################

# Predict Quality for Stanford Data (IMPRESSIONS): N=686
reports = sdata.copy()

# Not Diagnostic we only have 3, so eliminating
reports = reports[reports.quality != "Not Diagnostic"]
reports = reports[reports.quality.isnull()==False]
y = reports['quality'][reports.quality.isnull()==False]
impressions = reports['impression'][reports.quality.isnull()==False]
reports = reports['rad_report'][reports.quality.isnull()==False]
impression_result = predictPE(reports=impressions,y=y,labels=y.unique().tolist())
print_result(impression_result)

# scores:
# [ 0.85507246  0.84057971  0.85507246  0.88405797  0.91304348  0.94202899
#  0.85507246  0.89705882  0.91176471  0.91044776]


# confusion:
#            Diagnostic  Limited
# Diagnostic         586        3
# Limited             75       22


# cv:
# 10

report_result = predictPE(reports=reports,y=y,labels=y.unique().tolist())
print_result(report_result)


# scores:
# [ 0.85507246  0.85507246  0.85507246  0.86956522  0.86956522  0.84057971
#  0.85507246  0.86764706  0.86764706  0.86567164]


# confusion:
#            Diagnostic  Limited
# Diagnostic         585        4
# Limited             90        7


# cv:
# 10



# Predict Quality for Stanford Data (FULL REPORT):



# Predict Historicity for Stanford Data (IMPRESSIONS):



# Predict Historicity for Stanford Data (FULL REPORT):


###############################################################
# ROC CURVES For each, using one randomly sampled CV section-
# BETTER would be a completely new dataset
###############################################################

