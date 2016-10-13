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

def getMisclassified(reports,predictions,actual):
    '''getMisclassified will return misclassified reports based on vectors of predictions
    and actual labels
    :param reports: a list of reports associated with the labels
    :param predictions: a list of predicted labels
    :param actual: a list of actual labels
    '''
    # If reports is a series from a df, convert to list
    if isinstance(reports,list) == False:
        reports = reports.tolist()

    # Create list of tuples with report, prediction, actual if it's wrong
    wrongs = [(reports[x],predictions[x],actual[x]) for x in range(0,len(predictions)) if predictions[x] != actual[x]]
    if len(wrongs) > 0:
        wrongs = pandas.DataFrame(wrongs,columns=["report","prediction","actual"])
    else:
        wrongs = []
    return wrongs


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


def predictPE(reports,y,cv=10):
    '''predictPE will vectorize text and then run N fold cross validation 
    with an ensemble tree classifier to predict PE
    :param reports: should be a list or series of report text to be vectorized
    :param y: should be the actual rx labels (right now in Neg and Pos)
    :param labels: should be the unique labels (just for column/row names of confusion)
    '''

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
        clf.fit(trainX,trainy)
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
        wrongs = getMisclassified(reports=reports,
                                  predictions=predictions,
                                  actual=testy)

        # 5. Save sizes, etc.
        sizes = {"train":len(trainy),"test":len(testy)}

        # For each, let's save scores, cv, confusion, wrong
        result = {"scores":scores,
                  "cv":cv,
                  "confusion":confusion,
                  "importances":importances,
                  "misclassified":wrongs,
                  "N":sizes,
                  "params":clf.get_params()}

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


# Save all results to a dictionary
results = dict()

###############################################################
# PE PREDICTION [Neg,Pos]
###############################################################

PE = dict()

# Predict PE for Stanford Data (IMPRESSIONS):
reports = sdata['impression']
y = sdata['disease_state']
PE["stanford|disease_state|impression"] = predictPE(reports=reports,y=y)

# Predict PE for Stanford Data (FULL REPORTS):
reports = sdata['rad_report']
y = sdata['disease_state']
PE["stanford|disease_state|rad_report"] = predictPE(reports=reports,y=y)

# Predict PE for Chapman data:
chapman = cdata[cdata.disease_state.isnull()==False]
reports = chapman['impression'].tolist()

# Clean up text
getRidOf = ['[Report de-identified (Limited dataset compliant) by De-ID v.6.21.01.0]',
            'NULL']
reports = replace_text(reports,getRidOf,'')

y = chapman['disease_state']
PE["chapman|disease_state|impression"] = predictPE(reports=reports,y=y)


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
PE["stanford|disease_state_label|impression"] = predictPE(reports=reports,y=y)

reports = sdata['rad_report']
PE["stanford|disease_state_label|impression"] = predictPE(reports=reports,y=y)

results["pumonary_embolism"] = PE

###############################################################
# QUALITY PREDICTION [Diagnostic=589,Limited=97]
###############################################################

quality = dict()

# Predict Quality for Stanford Data:
reports = sdata.copy()

# Not Diagnostic we only have 3, so eliminating
reports = reports[reports.quality != "Not Diagnostic"]
reports = reports[reports.quality.isnull()==False]
y = reports['quality'][reports.quality.isnull()==False]

# IMPRESSIONS
impressions = reports['impression'][reports.quality.isnull()==False]
quality["stanford|quality|impression"] = predictPE(reports=impressions,y=y)

# FULL REPORT
reports = reports['rad_report'][reports.quality.isnull()==False]
quality["stanford|quality|rad_report"]= predictPE(reports=reports,y=y)
quality["stanford|quality|impression"] = predictPE(reports=impressions,y=y)

# Predict Quality for Chapman Data:

chapman = cdata[cdata.quality.isnull()==False]
reports = chapman['impression']

# Clean up text
getRidOf = ['[Report de-identified (Limited dataset compliant) by De-ID v.6.21.01.0]','NULL']
reports = replace_text(reports,getRidOf,'')
y = chapman['quality']
quality["chapman|quality|impression"] = predictPE(reports=reports,y=y)

# Save all quality results
results['quality'] = quality

###############################################################
# HISTORICITY PREDICTION
###############################################################

# Predict Historicity for Stanford Data:
hist = dict()

reports = sdata[sdata.historicity.isnull()==False]
y = reports['historicity']

# IMPRESSIONS
impressions = reports['impression']
hist["stanford|historicity|impression"] = predictPE(reports=impressions,y=y)

# FULL REPORT
reports = reports['rad_report']
hist["stanford|historicity|rad_report"] = predictPE(reports=impressions,y=y)

# Predict Historicity for Chapman data:
chapman = cdata[cdata.historicity.isnull()==False]

# We don't have enough samples for Mixed and No Consensus, just New and Old
reports = chapman['impression'][chapman.historicity.isin(['Mixed','No Consensus']) == False].tolist()

# Clean up text
getRidOf = ['[Report de-identified (Limited dataset compliant) by De-ID v.6.21.01.0]','NULL']
reports = replace_text(reports,getRidOf,'')
y = chapman['historicity'][chapman.historicity.isin(['Mixed','No Consensus']) == False]
hist["chapman|historicity|impression"] = predictPE(reports=reports,y=y)

# Save all hist results
results['historicity'] = hist

pickle.dump(results,open('results/chapman_stanford_results.pkl','wb'))

###############################################################
# ROC CURVES For each, using one randomly sampled CV section-
# BETTER would be a completely new dataset
###############################################################

