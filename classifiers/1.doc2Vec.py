#!/bin/python
# Note, this should be run within the Docker image provided to have all depdendencies, see
# ../Dockerfile in the base repo along with the README.md for setup instructions
# We will use gensim: https://radimrehurek.com/gensim/index.html

from wordfish.analysis import LabeledLineSentence, TrainSentences, DeepTextAnalyzer
from gensim.models import Doc2Vec
from gensim.models.doc2vec import LabeledSentence
from sklearn import linear_model
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing

import pandas
import pickle
import numpy
import os

# Let's make sure we are working from CODE HOME
CODE_HOME = os.environ["CODE_HOME"]
os.chdir(CODE_HOME)

# Read in data, this includes @mlungren annotations, filtered down to batches 1-4
data = pandas.read_csv("data/filtered_4_batches.tsv",sep="\t",index_col=0)
# there should be 944 rows

# Batches 1 and 2 are not independent, we will only use larger of the two
batches = [2.,3.,4.]
data = data[data.batch.isin(batches)]
# 691, if we use 1 instead we only have 470

# Create series of radiology reports
reports = data['rad_report']



### SUPPORTING FUNCTIONS ################################################################

class LabeledLineSentence(object):
    '''LabeledLineSentence is a general class will produce a labeled sentence 
     (iterable) to use for doc2vec input. In future will just be part of wordfish, testing for now
    '''
    def __init__(self,words_list,labels_list):
       self.words_list = words_list
       self.labels_list = labels_list
    def __iter__(self):
        for idx, words in enumerate(self.words_list):
            yield LabeledSentence(words=words,tags=[self.labels_list[idx]])


def get_vectors(model,words_list,labels):
    '''get_vectors will take an iterable of word lists (words_list) and labels
    and produce the words mapped onto the doc2vec space. If a list of words
    is not sufficiently long to produce a vector, it is filtered out
    '''
    da = DeepTextAnalyzer(model)
    vectors = pandas.DataFrame(columns=range(0,300))
    filtered_labels = []
    for idx,words in enumerate(words_list):
        vector = da.text2mean_vector(text=words,read_file=False)
        # Not enough overlap in the vocabulary means the vector is all NaN
        if vector != None:
            if vector.sum() != numpy.nan:
                vectors.loc[idx] = vector
                filtered_labels.append(labels.tolist()[idx])
    return {"labels":filtered_labels,"vectors":vectors}



def predict_linearRegression(X_train,y_train,X_test,Y_test):
    '''predict_linearRegression is scikit-learns out of the bag
    implementation of linear regression
    '''
    logreg = linear_model.LogisticRegression(C=1e5)
    logreg.fit(X_train, y_train)
    predictions = logreg.predict(X_test)
    confusion = pandas.DataFrame(confusion_matrix(Y_test, predictions,labels=list(lookup.keys())))
    confusion.index = list(lookup.keys())
    confusion.columns = list(lookup.keys())
    result = dict()
    return confusion    


####################################################
# Analysis 1: Use impression to classify PE 
#   - using elastic net trained on doc2vec vectors
####################################################


# Let's do holding out each batch, training on others
confusions = dict()

# In case we want to convert strings to numbers (not in use)
lookup = {"definitely negative":0.0, 
          "definitely positive":1.0,
          "Indeterminate":0.5,
          "probably negative":0.25,
          "probably positive":0.75}

print("Preparing to do linear regression models for impression reports...")

count=1
for remove_stop_words in [True,False]:
    for remove_non_english_chars in [True]:

        # For each set of params, we can take sum across batches of training and testing
        batchuid = "batch-%s-%s" %(int(remove_stop_words),int(remove_non_english_chars))
        print("Starting batch %s" %(batchuid))
        batchconfusions = dict()
        summed_confusion = pandas.DataFrame(0,columns=list(lookup.keys()),index=list(lookup.keys()))

        for holdout in batches:

            # Separate training and test data
            train_set = [x for x in batches if x != holdout]
            test_impression = data['impression'][data.batch==holdout]
            test_labels = data['disease_state_label'][data.batch==holdout]
            train_impression = data['impression'][data.batch.isin(train_set)]
            train_labels = data['disease_state_label'][data.batch.isin(train_set)]

            # Make some strings for pretty printing of labels
            training_ids = "|".join([str(int(x)) for x in train_set])
            testing_id = "%s" %(int(holdout))

            # Let's have a unique id so we can merge with whole report data later, eg 'holdout(2)-train(3|4)-stopw(1)-nonengrem(1)'
            uid = "holdout(%s)-train(%s)-rmstopw(%s)-rmnoneng(%s)" %(testing_id,                     # holdout id
                                                                     training_ids,                   # training ids joined with |
                                                                     int(remove_stop_words),         # 0/1
                                                                     int(remove_non_english_chars))  # 0/1

 
            print("RUNNING ANALYSIS %s:\n\ntrain(%s)\ntest(%s)\nrmstopw(%s)\nrmnoneng(%s)" %(count,training_ids,
                                                                                             testing_id,
                                                                                             remove_stop_words,
                                                                                             remove_non_english_chars))

            # Do the training
            words_list = TrainSentences(text_list=train_impression.tolist(),
                                        remove_non_english_chars=remove_non_english_chars,
                                        remove_stop_words=remove_stop_words)

            labeledDocs = LabeledLineSentence(words_list = words_list,
                                              labels_list = train_labels.tolist())


            # Build the vocabulary
            model = Doc2Vec(size=300,window=10,min_count=5,workers=11,alpha=0.025, min_alpha=0.025) # use fixed learning rate
            model.build_vocab(labeledDocs)

            for it in range(10):
                print("Training iteration %s" %(it))
                model.train(labeledDocs)
                model.alpha -= 0.002 # decrease the learning rate
                model.min_alpha = model.alpha # fix the learning rate, no decay
                model.train(labeledDocs)

            # Now let's create a mean filtered vectors for training data and testing data
            trainv = get_vectors(model,words_list,train_labels)
               # trainv['vectors']
               # trainv['labels']

            # len(trainv['labels'] is 198 - we lose just about 10 because of lack of overlap)

            # And mean vectors for testing
            test_words_list = TrainSentences(text_list=test_impression.tolist(),
                                             remove_non_english_chars=remove_non_english_chars,
                                             remove_stop_words=remove_stop_words)

            testv = get_vectors(model,test_words_list,test_labels)
            confusion = predict_linearRegression(X_train=trainv["vectors"],
                                                 y_train=trainv['labels'],
                                                 X_test=testv['vectors'],
                                                 Y_test=testv['labels'])
            count +=1    
            batchconfusions[uid] = confusion
            summed_confusion += confusion
        
        # When we finish a set of holdout/training (a batch), add summed and normalized version
        batchconfusions['sum-%s' %(batchuid)] = summed_confusion
        total_confusions = summed_confusion.sum().sum()
        batchconfusions['norm-%s' %(batchuid)] = summed_confusion / total_confusions 
        confusions[batchuid] = batchconfusions

        # save as we go!
        pickle.dump(confusions,open("data/impression_lr_confusions.pkl","wb"))
