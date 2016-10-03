# Prediction of Pulmonary Embolism

## Background

### PE Finder: a Rule-Based Classifier
Brian Chapman created an open source "PE Finder" that would use rule-based learning to automatically detect pulmonary embolism in radiology reports. He did this using the "Impression" portion of the report. They used software called "ehose" to annotate the reports at the entity level (individual words).

### What do we want to do?
We want to use machine learning to try to improve upon the performance of rule-based learning. 

- we tried annotating at the word level, the performance wasn't an improvement over document level, and so we are annotating at the document level.
- we have the original Chapman data (references to "utah") and likely will ultimately want to compare the two methods using this data (e.g., Chapman vs. our method on Stanford data, and Chapman vs. our method on Chapman data)

### What have we done
The Stanford data has been annotated in "batches," 4 so far.
- The first two (as described above) were entity level annotations using ehose, this was found to not help, but it's obviously a lot more work (and not worth it).
- The second two were on the document level. This data (NOT SHARED in this repo, but included in a local "data" folder) is a `.csv` file that will be described in further detail in this document.

### Collaborations
- Robin Ball: was mentioned to be a contact at Stride. She wants to take results from classifier data and combine with pathology data to produce a better classifer for PE.

A larger goal is to generate a dataset that we can do deep learning on radiology reports. Aka, we produce probabilties, and the probabilities are features for others analyses.

### Improvements and Things to Watch out For
- The optimization is being done on accuracy (this was how Chapman did it) however this can be improved by using AUC (area under the curve) or F-statistic.
- Batches 1 and 2 are NOT independent. This means we cannot train on 2 and then test on 1. 2 is a biased sample, and it can be good for training, as long as 1 is not the test.
- finite population correction factor: Yu mentioned this is something like 0.01*population size. The general idea is that if the population is finite, if we take a sample that is too large it becomes biased. For this sample, this number is around ~5-6K. We can likely do one more batch and do ok, but more than that, we need to take this into account.

## Implementation/Model Notes
Yu used:

      sklearn.feature_extraction.text.CountVectorizer


If we aim to ultimately compare classifiers over data sets, the recommended methods from Yu are as follows:

      1.Demsar, J. (2006). Statistical comparisons of classifiers over multiple data sets. Journal of Machine Learning Research, 7, 1–30.

      1.T. G. Dietterich. Approximate statistical tests for comparing supervised classification learning al- gorithms. Neural Computation, 10:1895–1924, 1998. 

      Wilcoxon signed-ranks test for 2 classifiers over multiple sets
      1.Friedman test followed by post-hoc Hommel test for multiple classifiers over multiple sets

      McNemar Test for 2 classifiers over 1 set
