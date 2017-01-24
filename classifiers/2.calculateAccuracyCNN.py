#!/bin/python

import pandas as pd
from utils import makeConfusion, calculateMetrics


################################################################
# Stanford data (N=944) classified using PE Finder
# How many Stanford cases did PEFinger get right?
################################################################


# makeConfusion(y_true,y_pred,labels) 
# labels is the set of possible labels

# How many did we get right?
sdata = pd.read_csv('results/stanford-cnn-results.csv')
y_true = sdata['disease_state'].tolist()
y_pred = sdata['pe_rslt'].apply(lambda x : 'Pos' if x == 1 else 'Neg')
print(makeConfusion(y_true,y_pred,["Neg","Pos"]))
print('Accuracy: %f Precision: %f Recall: %f'%calculateMetrics(y_true, y_pred))

#      Neg  Pos
# Neg  789   41
# Pos    4  110
# Accuracy: 0.952331 Precision: 0.728477 Recall: 0.964912


################################################################
# Chapman data (N=860)-1 nan classified using PE Finder
# How many cases did PEFinger get right?
################################################################


cdata = pd.read_csv('results/chapman-cnn-results.csv')
cdata = cdata[cdata['disease_state'].isnull()==False] #one case
y_true = cdata['disease_state'].tolist()
y_pred = cdata['pe_rslt'].apply(lambda x : 'Pos' if x == 1 else 'Neg')
print(makeConfusion(y_true,y_pred,["Neg","Pos"]))
print('Accuracy: %f Precision: %f Recall: %f'%calculateMetrics(y_true, y_pred))

#      Neg  Pos
# Neg  500   66
# Pos   15  278
# Accuracy: 0.905704 Precision: 0.808140 Recall: 0.948805

