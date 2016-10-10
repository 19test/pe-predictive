# Predicting PE from Radiology Reports

This will be a set of preliminary analyses for me to get familiar with the data, described in the [previous README](../README.md). If you did not read this previous document, know that the entire analysis (I ran) from within a Docker image to deal with installation of all dependencies, and the final command to use the image was:

      docker run -it vanessa/pe-predictive

You can of course install on your local machine, but then you will need to install dependencies (not recommended).

## Overview
This small analysis will first convert the radiology reports/impressions to a vector representation using [doc2vec](http://radimrehurek.com/gensim/models/doc2vec.html), and then use logistic regression on training and holdout (test) sets to assess the predictive ability of the reports/impressions. I chose doc2vec for several reasons:

## Reasons to try doc2vec
- the framework allows for a document of any length to be mapped into the space.
- a model can be saved and updated

## Analysis steps

### Filter and preparing data

**0.reportsFilter.py**
The script [0.reportsFilter.py](0.reportsFilter.py) simply loads the data (from what I have, the `final_3.csv`). It summarizes counts for each of the class labels, along with columns provided and shows the change in size before and after filtering. The final task is to save a filtered dataset from the raw data, which is `../data/filtered_reports.tsv` (not provided in this repo).

**1.doc2vec.py**
The script [1.doc2Vec.py](1.doc2Vec.py) (mostly self documented) uses logistic regression on the doc2vec vectors to see if we can distinguish PE / not PE. The performance was terrible (context of the words is not the key to figuring out the diagnosis, likely) but I thought it was interesting that using the whole report had better performance than just the impression section. Specifically, when we add up the diagonal of the normalized confusion matrices (the sum of the ones we got right, each representing a percentage of all the reports):

Here is logistic regression for removing and not removing stop words:
 
### Impressions

	# numpy.trace(confusions['batch-1-1']['norm-batch-1-1'])
	# 0.60207100591715978

	# numpy.trace(confusions['batch-0-1']['norm-batch-0-1'])
	# 0.59334298118668594

	# Ouch! Does not work!


### Entire reports


	# numpy.trace(confusions['batch-1-1']['norm-batch-1-1'])
	# 0.6758321273516642

	# numpy.trace(confusions['batch-0-1']['norm-batch-0-1'])
	# 0.66714905933429813


It is interesting that we do better when we have the entire report. We probably would want to first start from what Yu was doing (I'm not totally sure, @mlungren will hopefully give insight), and then try to improve upon that. What we can do is different visualizations (both of data and of misclassified cases) to improve some classifier.

**2.countVectorizer.py**

The script [2.countVectorizer.py](2.countVectorizer.py) (also mostly self documented) uses the scikit learn count vectorizer to build ensemble tree classifiers for each of the same holdout groups. There were much better results for this method (and I believe that I reproduced the original result) however it was very sensitive to the data used for train and test:

### Impressions

	holdout(3)-train(2|4)
	Accuracy: 0.824561403509

	holdout(2)-train(3|4)
	Accuracy: 0.915611814346

	holdout(4)-train(2|3)
	Accuracy: 0.74375


### Entire Reports


	holdout(3)-train(2|4)
	Accuracy: 0.842105263158

	holdout(2)-train(3|4)
	Accuracy: 0.909282700422

	holdout(4)-train(2|3)
	Accuracy: 0.73125


Using 2|3 to train and 4 to test has equivalent results between using the entire report and just impressions. There is some improvement in using the full report when using 3 to test, and slight worse performance for full reports when using set 2 to test. Likely if we did these many times, we would see there is some variance (and the two aren't significantly different), but I have not yet tested this.


## Long Term Goals
- Obviously, build a better machine learning classifier
- It is not adequate to start with an already pre-processed data file - we will talk about the entire pipeline from data collection through the end of analysis. We want to be able to acquire new data and feed it seamlessly into this pipeline with minimal manual pain.


## Questions I have
- Yu mentioned that the batches could be used in sets for training/testing, and not to use 1 and 2 as they are not independent of one another. My larger question is why the data should be separated into batches to begin with? In other words, why not just combine data that is not redundant (sets 2,3,4) and then do 10 fold cross validation? 
- The doc2vec can also produce mean vectors for groups of things - we could probably make a classifier to predict larger things about the report.
