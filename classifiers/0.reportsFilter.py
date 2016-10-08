#!/bin/python

# Note, this should be run within the Docker image provided to have all depdendencies, see
# ../Dockerfile in the base repo along with the README.md for setup instructions

import pandas
import os

# Let's make sure we are working from CODE HOME
CODE_HOME = os.environ["CODE_HOME"]
os.chdir(CODE_HOME)

# First load the data - we have final_3.tsv from Yu, and ideally we should have a pipeline
# that goes from data collection --> input into this algorithm. This will work for now.
rawData = pandas.read_csv("data/final_3.csv")

#rawData.shape
# (117816, 19)

#rawData.columns
# ['pat_deid', 'order_deid', 'days_age_at_ct', 'rad_report', 'impression',
#       'batch', 'disease_state_label', 'uncertainty_label', 'quality_label',
#       'historicity_label', 'disease_state_prob', 'uncertainty_prob',
#       'quality_prob', 'historicity_prob', 'disease_PEfinder',
#       'looking_for_PE?', 'train=2/test=1', 'disease_probability_test',
#       'probability_looking_for_PE'],

# We don't care about the latter columns that were built from a previous model - let's filter
# down to @mlungren's annotations, and the original reports

rawData = rawData.drop(labels=["disease_state_prob","uncertainty_prob","quality_prob","historicity_prob",
                               "train=2/test=1","disease_probability_test","probability_looking_for_PE"],axis=1)


# How many labels do we have?
rawData.disease_state_label.count()

rawData.disease_state_label.value_counts()

#definitely negative    793
#definitely positive    108
#probably negative       23
#Indeterminate           14
#probably positive        6
#Name: disease_state_label, dtype: int64

# Let's filter down to those that are in a batch (meaning we can ues them)
data = rawData[rawData.disease_state_label.isnull()==False]

# How many in each batch?
data.batch.value_counts()
#2.0    474
#1.0    253
#4.0    160
#3.0     57
#Name: batch, dtype: int64

# Batch 2 has the most, but we know they are not independent from batch 1, so we should use one
# or the other.
data.to_csv("data/filtered_4_batches.tsv",sep="\t")
