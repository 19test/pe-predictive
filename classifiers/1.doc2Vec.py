#!/bin/python
# Note, this should be run within the Docker image provided to have all depdendencies, see
# ../Dockerfile in the base repo along with the README.md for setup instructions
# We will use gensim: https://radimrehurek.com/gensim/index.html
# doc2vec architecture: the corresponding algorithms are “distributed memory” (dm) and “distributed bag of words” (dbow). Since the distributed memory model performed noticeably better in the paper, that algorithm is the default when running Doc2Vec.

from wordfish.analysis import LabeledLineSentence, TrainSentences
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
            # This is a list of words, and a list of labels
            # vectors for labels AND words are learned simeotaneously
            # our use case is to give unique label per example, but you could have common labels
            yield LabeledSentence(words,["%s_%s" %(self.labels_list[idx],idx)])


def get_vectors(model,words_list,labels):
    '''get_vectors will take an iterable of word lists (words_list) and labels
    and produce the words mapped onto the doc2vec space. If a list of words
    is not sufficiently long to produce a vector, it is filtered out
    '''
    allLabels = list(model.docvecs.doctags.keys()):
    vectors = pandas.DataFrame(model.docvecs.doctag_syn0norm) 
    vectors.index = allLabels   
    trainOrtest = [x.split("_")[0] for x in labels]
    return {"labels":trainOrtest,"vectors":vectors}


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
# Setup for Analysis 1: Use impression to classify PE 
#   - parameters used and left as default
####################################################


# Let's do holding out each batch, training on others
confusions = dict()

# ANALYSIS PARAMETERS
size=300          # The size of the feature vector will be 300
window=10         # max distance between the predicted word and context words used for prediction within a document
min_count=5       # ignore all words with total frequency lower than this.
workers=11        # how many workers used to train the model (parallelization)
alpha=0.025       # initial learning rate (will linearly drop to zero as training progresses).
min_alpha=alpha   # except... we are setting at a fixed value :)
iters=10          # iterations used to tweak final alpha

# Parameters not set
# dm defines the training algorithm. By default (dm=1), ‘distributed memory’ (PV-DM) is used (performed best in paper). Otherwise, distributed bag of words (PV-DBOW) is employed.

# seed:     
# for the random number generator. Note that for a fully deterministically-reproducible run, you must also limit the 
# model to a single worker thread, to eliminate ordering jitter from OS thread scheduling. (In Python 3, 
# reproducibility between interpreter launches also requires use of the PYTHONHASHSEED environment variable to control 
# hash randomization.

# max_vocab_size:
# limit RAM during vocabulary building; if there are more unique words than this, then prune the infrequent ones. 
# Every 10 million word types need about 1GB of RAM. Set to None for no limit (default).

# sample: 
# threshold for configuring which higher-frequency words are randomly downsampled;
# default is 0 (off), useful value is 1e-5.

# iter = number of iterations (epochs) over the corpus. The default inherited from Word2Vec is 5, but values of 10 or 20 are 
# common in published ‘Paragraph Vector’ experiments.

# hs:
# if 1 (default), hierarchical sampling will be used for model training (else set to 0).

# negative:
# if > 0, negative sampling will be used, the int for negative specifies how many “noise words” should be drawn 
# (usually between 5-20)

# dm_mean:
# if 0 (default), use the sum of the context word vectors. If 1, use the mean. Only applies when dm is used in non-
# concatenative mode.

# dm_concat:
# if 1, use concatenation of context vectors rather than sum/average; default is 0 (off). Note concatenation 
# results in a much-larger model, as the input is no longer the size of one (sampled or arithmatically combined) word 
# vector, but the size of the tag(s) and all words in the context strung together.

# dm_tag_count:
# expected constant number of document tags per document, when using dm_concat mode; default is 1.

# dbow_words 
# if set to 1 trains word-vectors (in skip-gram fashion) simultaneous with DBOW doc-vector training; default is 0 
# (faster training of doc-vectors only).

# trim_rule:
# vocabulary trimming rule, specifies whether certain words should remain
# in the vocabulary, be trimmed away, or handled using the default (discard if word count < min_count). Can be None # 
# (min_count will be used), or a callable that accepts parameters (word, count, min_count) and returns either 
# util.RULE_DISCARD, util.RULE_KEEP or util.RULE_DEFAULT. Note: The rule, if given, is only used prune vocabulary during 
# build_vocab() and is not stored as part


####################################################
# Analysis 1: Use impression to classify PE 
#   - using logistic regression on doc2vec vectors
####################################################


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

            # Add train or test to labels, so we can distinguish later
            train_labels = ["TRAIN_%s" %(x) for x in train_labels.tolist()]
            test_labels = ["TEST_%s" %(x) for x in test_labels.tolist()]

            # Compile them together
            allLabels = train_labels + test_labels
            allImpression = train_impression.tolist() + test_impression.tolist()

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
            words_list = TrainSentences(text_list=allImpression,
                                        remove_non_english_chars=remove_non_english_chars,
                                        remove_stop_words=remove_stop_words)

            labeledDocs = LabeledLineSentence(words_list = words_list,
                                              labels_list = allLabels)


            # Build the vocabulary
            model = Doc2Vec(size=size,window=window,
                            min_count=min_count,
                            workers=workers,
                            alpha=alpha,
                            min_alpha=min_alpha) # use fixed learning rate

            # train_words=True
            # train_lbls=True

            # Build the vocabularity and fine tune the alpha (manually control learning rate over 10 epochs)
            model.build_vocab(labeledDocs)
            for it in range(iters):
                print("Training iteration %s" %(it))
                model.train(labeledDocs)
                model.alpha -= 0.002          # decrease the learning rate
                model.min_alpha = model.alpha # fix the learning rate, no decay
                model.train(labeledDocs)

            # Now let's create a data frame with training and testing data
            df = get_vectors(model,words_list,allLabels)
            # df['vectors']
            # df['labels']

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
