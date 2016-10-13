#!/env/python

import radnlp

import pyConTextNLP.pyConTextGraph as pyConText
import pyConTextNLP.itemData as itemData

import os
import pandas
import radnlp.io  as rio
import radnlp.view as rview
import radnlp.rules as rules
import radnlp.schema as schema
import radnlp.utils as utils
import radnlp.split as split
import radnlp.classifier as classifier

import radnlp.utils as utils
from radnlp.data import classrslts

from sklearn.metrics import confusion_matrix

### SUPPORTING FUNCTIONS ################################################################

def makeConfusion(y_true,y_pred,labels):
    '''makeConfusion simply wraps scikit-learns confusion_matrix function,
    and turns into a pandas DataFrame with column and row names
    '''
    confusion = confusion_matrix(y_true, y_pred)
    confusion = pandas.DataFrame(confusion,columns=labels,index=labels)
    return confusion


def getRx(pe_result_series,positive_label=None):
    '''getRx will get the label for a result from pe finder
    '''
    if positive_label == None:
        positive_label = "pulmonary_embolism"

    rx = []
    for res in pe_result_series.tolist():
        labels = list(res.classification_result.keys())
        # No label assignment considered negative result
        if len(labels) == 0:
            rx.append("Neg")
        else:
            # Confirmed with Stanford data that there is only one label, or none 
            if labels[0] == positive_label:
                rx.append("Pos")
            else:
                rx.append("Neg")
    return rx
    

### CHAPMAN FUNCTIONS ###################################################################

def analyze_report(report, modifiers, targets, rules, schema):
    """
    given an individual radiology report, creates a pyConTextGraph
    object that contains the context markup
    report: a text string containing the radiology reports
    """
    markup = utils.mark_report(split.get_sentences(report),
                         modifiers,
                         targets)
    
    clssfy = classifier.classify_document_targets(markup,
                                                  rules[0],
                                                  rules[1],
                                                  rules[2],
                                                  schema)
    return classrslts(context_document=markup, exam_type="ctpa", report_text=report, classification_result=clssfy)
