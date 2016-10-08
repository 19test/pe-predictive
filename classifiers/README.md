# Predicting PE from Radiology Reports

This will be a set of preliminary analyses for me to get familiar with the data, described in the [previous README](../README.md). If you did not read this previous document, know that the entire analysis (I ran) from within a Docker image to deal with installation of all dependencies, and the final command to use the image was:

      docker run -it vanessa/pe-predictive

The image (should not be) on Docker Hub, and runnable without needing to build it from this repo.

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
The script [1.doc2vec.py](1.doc2vec.py) has the following sections:


## Improvements/Thinking
- It is not adequate to start with an already pre-processed data file - we will talk about the entire pipeline from data collection through the end of analysis. We want to be able to acquire new data and feed it seamlessly into this pipeline with minimal manual pain.

-    # NOTE: we could also look at the vectors for the groups above, might be interesting


## Questions I have
- Yu mentioned that the batches could be used in sets for training/testing, and not to use 1 and 2 as they are not independent of one another. My larger question is why the data should be separated into batches to begin with? In other words, why not just combine data that is not redundant (sets 2,3,4) and then do 10 fold cross validation? 
