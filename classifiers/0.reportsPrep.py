#!/bin/python

# Note, this should be run within the Docker image provided to have all depdendencies, see
# ../Dockerfile in the base repo along with the README.md for setup instructions

import pandas
import numpy
import os

# Let's make sure we are working from CODE HOME
CODE_HOME = os.environ["CODE_HOME"]
os.chdir(CODE_HOME)

# First load the data - we have final_3.tsv from Yu, and ideally we should have a pipeline
# that goes from data collection --> input into this algorithm. This will work for now.
rawData = pandas.read_csv("stanford-data/final_3.csv")

#rawData.shape
# (117816, 19)

# We want the Stanford data to have the exact same form as the chapman data, meaning
# these columns
# 'id', 'impression', 'disease_state', 'uncertainty', 'quality', 'historicity', 'pe rslt'], dtype='object'

# Here is what we have:
#rawData.columns
# ['pat_deid', 'order_deid', 'days_age_at_ct', 'rad_report', 'impression',
#  'batch', 'disease_state_label', 'uncertainty_label', 'quality_label',
#  'historicity_label', 'disease_state_prob', 'uncertainty_prob',
#  'quality_prob', 'historicity_prob', 'disease_PEfinder',
#  'looking_for_PE?', 'train=2/test=1', 'disease_probability_test',
#  'probability_looking_for_PE'],

# From Chapman paper:
# ...probably positive and definitely positive were collapsed to positive; probably negative, indeterminate, and definitely negative were considered negative;

# Our disease_state_labels should be mapped onto Neg, Pos, and nan
lookup = {"definitely negative":"Neg", 
          "definitely positive":"Pos",
          "probably negative":"Neg",
          "probably positive":"Pos",
          "Indeterminate":"Neg",
          numpy.nan:numpy.nan}

disease_labels = [lookup[x] for x in rawData.disease_state_label.tolist()]
rawData['disease_state'] = disease_labels

# Same with uncertainty labels
# From Chapman paper:
# definitely negative and definitely positive were considered certain; and probably negative, inderminate, and probably positive were considered uncertain.
lookup = {"definitely negative":"Yes", 
          "definitely positive":"Yes",
          "probably negative":"No",
          "probably positive":"No",
          "Indeterminate":"No",
          numpy.nan:numpy.nan}

uncertain_labels = [lookup[x] for x in rawData.uncertainty_label.tolist()]
rawData['uncertainty'] = uncertain_labels


# Quality label
lookup = {"diagnostic":"Diagnostic", 
          "non-diagnostic":"Not Diagnostic",
          "limited":"Limited",
          numpy.nan:numpy.nan}

quality_labels = [lookup[x] for x in rawData.quality_label.tolist()]
rawData['quality'] = quality_labels


# Historicity labels
lookup = {"new":"New", 
          "old":"Old",
          "mixed":"Mixed",
          numpy.nan:numpy.nan}

hist_labels = [lookup[x] for x in rawData.historicity_label.tolist()]
rawData['historicity'] = hist_labels

# We don't care about the latter columns that were built from a previous model - let's filter
# down to @mlungren's annotations, and the original reports

rawData = rawData.drop(labels=["disease_state_prob","uncertainty_prob","quality_prob","historicity_prob",
                               "train=2/test=1","disease_probability_test","probability_looking_for_PE"],axis=1)

# Let's filter down to those that are in a batch (meaning we can ues them)
rawData = rawData[rawData.disease_state.isnull()==False]

# How many in each batch?
rawData.batch.value_counts()
#2.0    474
#1.0    253
#4.0    160
#3.0     57
#Name: batch, dtype: int64

# Batch 2 has the most, but we know they are not independent from batch 1, so we should use one
# or the other.
rawData.to_csv("stanford-data/stanford_df.tsv",sep="\t")
