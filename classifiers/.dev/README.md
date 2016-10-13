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
