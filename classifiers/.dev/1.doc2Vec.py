#!/bin/python
# Note, this should be run within the Docker image provided to have all depdendencies, see
# ../Dockerfile in the base repo along with the README.md for setup instructions
# We will use gensim: https://radimrehurek.com/gensim/index.html
# doc2vec architecture: the corresponding algorithms are “distributed memory” (dm) and “distributed bag of words” (dbow). Since the distributed memory model performed noticeably better in the paper, that algorithm is the default when running Doc2Vec.

from wordfish.analysis import TrainSentences
from gensim.models import Doc2Vec
from gensim.models.doc2vec import LabeledSentence
from sklearn import linear_model
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing

import pandas
import pickle
import numpy
import os
import re

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



### SUPPORTING FUNCTIONS ################################################################

class LabeledLineSentence(object):
    '''LabeledLineSentence is a general class will produce a labeled sentence 
     (iterable) to use for doc2vec input. In future will just be part of wordfish, testing for now
    :param words_list: an iterable to yield a lists of words, one list per report
    :param labels_list: a list of labels corresponding to the pe status (not the patient id)
    :param uid_list: the uid of the patients, as index for the data frames
    '''
    def __init__(self,words_list,labels_df):
       self.words_list = words_list
       self.labels_df = labels_df
    def __iter__(self):
        for idx, words in enumerate(self.words_list):
            # This is a list of words, and a list of tags
            # vectors for labels AND words are learned simeotaneously
            # our use case is to give unique label per example, but I've added "groups" as well
            uid = self.labels_df.index.tolist()[idx]
            group_label = self.labels_df.loc[uid]["CLASS"]
            disease_label = self.labels_df.loc[uid]["disease_state_label"]
            tags = [uid,group_label,disease_label]
            # [302105833, 'TRAIN', 'definitely negative']
            yield LabeledSentence(words,tags)
            # We add tags for the document labels, whether train or test, and the disease label
            # LabeledSentence([words],[tags])


def get_vectors(model,words_list,labels):
    '''get_vectors will take an iterable of word lists (words_list) and labels
    and produce the words mapped onto the doc2vec space. If a list of words
    is not sufficiently long to produce a vector, it is filtered out
    :param model: the doc2vec model
    :param labels: a dataframe with (index) as uid, and columns CLASS and disease_state_label
    '''
    # We want data organized into all, train, and test
    vecs = dict()

    # First prepare data for all, combined train and test
    vectors_index = list(model.docvecs.doctags.keys()) # included we have labels of type:
    vectors = pandas.DataFrame(model.docvecs.doctag_syn0)

    # Filter vectors down to just the individual records (eg, remove:
    # ['TRAIN',
    #'Indeterminate',
    #'TEST',
    #'probably negative',
    #'definitely negative',
    #'probably positive',
    #'definitely positive'])

    vectors.index = vectors_index
    vectors = vectors.loc[labels.index]
    # NOTE: we could also look at the vectors for the groups above, might be interesting
    
    # Some reports (~5 in testing) don't have vectors if they don't meet minimum size
    vectors = vectors[vectors.isnull().any(axis=1)==False]
    labels = labels[labels.index.isin(vectors.index)]

    # All data
    vecs['all'] = {"labels":labels,
                   "vectors":vectors}

    # Now for each of train and test
    train_vectors = vectors.loc[labels.index[labels.CLASS=="TRAIN"]]
    test_vectors = vectors.loc[labels.index[labels.CLASS=="TEST"]]
    trainLabels = labels[labels.CLASS=="TRAIN"]
    testLabels = labels[labels.CLASS=="TEST"]

    # Train data
    vecs['train'] = {"labels":trainLabels,
                   "vectors":train_vectors} 

    vecs['test'] = {"labels":testLabels,
                   "vectors":test_vectors} 

    return vecs


def predict_logisticRegression(train,test):
    '''predict_logisticRegression is scikit-learns out of the bag
    implementation of logistic regression
    :param train: a dict with keys vectors, labels (CLASS, disease_state_label) and pat_id are index
    :param test: ditto for test
    '''
    logreg = linear_model.LogisticRegression(C=1e5)
    logreg.fit(train['vectors'],train['labels']['disease_state_label'].tolist())
    predictions = logreg.predict(test['vectors'])
    confusion = pandas.DataFrame(confusion_matrix(test['labels']['disease_state_label'].tolist(), 
                                                 predictions,
                                                 labels=list(lookup.keys())))
    confusion.index = list(lookup.keys())
    confusion.columns = list(lookup.keys())
    # Eventually we can look here to see what we got wrong
    return confusion    


def predictPE(inputDataLabel,model_type):
    '''predictPE is a massive wrapper to run all iterations of training and testing using some input label.
    (eg, impression or report). A dictionary structure of confusion matrices is returned, include a summed
    and normalized version to represent the success of the entire batch. Note that this function is dependent
    on some of the global variables defined above (not proper, but will work for this batch script :))
    :param model_type: right now I just tried "logistic_regression"
    :param inputDataLabel: should be one of "impression" or "rad_report"
    '''
    confusions = dict()
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
                # Question - is there any reason to split via batches? Bias in this?
                train_set = [x for x in batches if x != holdout]
                test_impression = data[inputDataLabel][data.batch==holdout]
                test_labels = pandas.DataFrame(data['disease_state_label'][data.batch==holdout])
                test_ids = data['order_deid'][data.batch==holdout]
                train_impression = data[inputDataLabel][data.batch.isin(train_set)]
                train_labels = pandas.DataFrame(data['disease_state_label'][data.batch.isin(train_set)])
                train_ids = data['order_deid'][data.batch.isin(train_set)]

                train_labels["CLASS"] = "TRAIN"
                test_labels["CLASS"] = "TEST"
                allIds = train_ids.append(test_ids).tolist()
                allLabels = train_labels.append(test_labels)
                allLabels.index = allIds            

                # Compile them together
                allImpression = train_impression.tolist() + test_impression.tolist()
                # sanity check
                assert(len(allIds) == len(allImpression) == len(allLabels))

                # Make some strings for pretty printing of train/test batch
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
                                                  labels_df = allLabels)

                # Build the vocabulary
                model = Doc2Vec(size=size,
                                window=window,
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

                # This was done manually during testing, for impressions
                #model.save('data/model.doc2vec')
                
                # Now let's create an object with data frames with training and testing data and labels
                vecs = get_vectors(model=model,
                                   words_list=words_list,
                                   labels=allLabels)
                # df['all']   ....
                # df['train'] ....
                # df['test']  .... ['labels'] <-- df with columns disease_state_label,CLASS, and index as patid
                #                  ['vectors'] <-- index is also patid

                if model_type == "logistic_regression":
                    confusion = predict_logisticRegression(train=vecs['train'],
                                                           test=vecs['test'])
                count +=1    
                batchconfusions[uid] = confusion
                summed_confusion += confusion
        
            # When we finish a set of holdout/training (a batch), add summed and normalized version
            batchconfusions['sum-%s' %(batchuid)] = summed_confusion
            total_confusions = summed_confusion.sum().sum()
            batchconfusions['norm-%s' %(batchuid)] = summed_confusion / total_confusions 
            confusions[batchuid] = batchconfusions

    return confusions


####################################################
# Setup for Analysis 1: Use impression to classify PE 
#   - parameters used and left as default
####################################################


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

print("Preparing to do logistic regression models for impression reports...")

confusions = predictPE(inputDataLabel='impression',
                       model_type='logistic_regression')

pickle.dump(confusions,open("data/impression_lr_confusions.pkl","wb"))
# numpy.trace(confusions['batch-1-1']['norm-batch-1-1'])
# 0.60207100591715978

# numpy.trace(confusions['batch-0-1']['norm-batch-0-1'])
# 0.59334298118668594

# Ouch! Does not work!


####################################################
# Analysis 2: Use reports to classify PE 
#   - using logistic regression on doc2vec vectors
####################################################

### LOGISTIC REGRESSION

print("Preparing to do logistic regression models for entirety of reports...")

confusions = predictPE(inputDataLabel='rad_report',
                       model_type='logistic_regression')
pickle.dump(confusions,open("data/report_lr_confusions.pkl","wb"))
# numpy.trace(confusions['batch-1-1']['norm-batch-1-1'])
# 0.6758321273516642

# numpy.trace(confusions['batch-0-1']['norm-batch-0-1'])
# 0.66714905933429813
# interesting! We do better when we have the whole report.
# This sort of makes sense for doc2vec



####################################################
# Analysis 3: Compare results with PEFinder 
####################################################

# Not worth doing this, we did terrible!
peFinder = data["disease_PEfinder"]
