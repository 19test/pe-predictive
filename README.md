# Prediction of Pulmonary Embolism

## Folder Organization

- [chapman](chapman): includes reproduction of Chapman PE prediction methods.

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
- finite population correction factor: Yu mentioned this is something like 0.05*population size. The general idea is that if the population is finite, if we take a sample that is too large it becomes biased. For this sample, this number is around ~5-6K. We can likely do one more batch and do ok, but more than that, we need to take this into account.

## Yu's Implementation/Model Notes
Yu used:

      sklearn.feature_extraction.text.CountVectorizer


If we aim to ultimately compare classifiers over data sets, the recommended methods from Yu are as follows:

      1.Demsar, J. (2006). Statistical comparisons of classifiers over multiple data sets. Journal of Machine Learning Research, 7, 1–30.

      1.T. G. Dietterich. Approximate statistical tests for comparing supervised classification learning al- gorithms. Neural Computation, 10:1895–1924, 1998. 

      Wilcoxon signed-ranks test for 2 classifiers over multiple sets
      1.Friedman test followed by post-hoc Hommel test for multiple classifiers over multiple sets

      McNemar Test for 2 classifiers over 1 set

## Vanessa's Implementation / Modes Notes
The environment can be built for the scripts in [classifiers](classifiers) using Docker:

      docker build -t vanessa/pe-predictive .

Then run and shell into the image:

      docker run -it vanessa/pe-predictive bash

If you want to have data also appear on your local machine somewhere, be sure to map a volume from the container:

      docker run -it -v results:/tmp vanessa/pe-predictive

The working directory will be the folder `/code` and within this folder you will see the same files as on your local machine. All python dependencies are installed, and `python3` is aliased to the `python` command, and `ipython3` to `ipython`, and `pip3` to `pip`. Finally, an environmental variable called `CODE_HOME` is set to make sure we don't have path errors in our scripts.

For details on the code, see the [README.md](classifiers/README.md) included with it.

If you forget to add a volume when you run the container (oups) you can copy data from the container to your local machine:

      docker cp 74feea5941b5:/code/data/filtered_4_batches.tsv $PWD/data
      docker cp 74feea5941b5:/code/data/impression_lr_confusions.pkl $PWD/data

Where the id of the container is obtained with `docker -ps`

## Data Notes

### Organization
The data_*.csv files are currently being used for @mlungren to randomly select and annotate. We need to figure out a different/better way for this task, likely starting with where the data originates, and what the intended use is. Keeping the classiifer results, and the raw data (reports), and the other various meta data in one massive file makes me very anxious and is not a suitable long term solution for this kind of work.

### Column Values
This is my best understanding of the meta-data fields:

- pat_deid: this is a deidentified patient ID. I would like to eventually know where and how this is generated, and where it links to, but this level of understanding is suitable for now.
- order_deid: is the id of the order, the idea being that one patient could have multiple orders. Question: what if a patient has more than one, if the reports are different is the idea that it's not an issue?
- rad_report: this is literally the entire radiologist report in a column.
- impression: this is an extracted portion of the report. We should note this is done programatically, and while probably most of them are OK, there could be a subset with errors.
- batch: is the batch number mentioned above. There are currently 4.
- disease_state_label, 
- uncertainty_label, 
- quality_label, 
- historicity_label: these are manually labeled annotations by @mlungren
- disease_state_prob,
- uncertainty_prob,
- quality_prob,
- historicity_prob: these are produced by Yu's classifier. The code is (somewhere) in ipython notebooks.
- disease_PEfinder: is the PEfinder (Chapman) being run on these datasets. The accuracy of this has not been assessed, but this would be useful for some future paper.
- looking_for_PE?: Was the purpose of the report to look for PE (1), or was it an indicental finding (0).
- train=2/test=1: This is a column to indicate that some of the records (2) were used for training, and some for testing (1).
- disease_probability_test: this is the outcome of the model building with the labels specified by train=2/test=1
- probability_looking_for_PE: Another of Yu's models to predict if the exam was done looking for PE, Yu noted this performed very well (i.e., we can predict if the assessment was done to specifically look for PE based on the report alone)
 

