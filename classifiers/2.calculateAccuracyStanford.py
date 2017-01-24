#!/bin/python

import pickle
from utils import makeConfusion, getRx, calculateMetrics

################################################################
# Stanford data (N=944) classified using PE Finder
# How many Stanford cases did PEFinger get right?
################################################################


# makeConfusion(y_true,y_pred,labels) 
# labels is the set of possible labels

# How many did we get right?
sdata = pickle.load(open('results/stanford-pe-results.pkl','rb'))
y_true = sdata['disease_state'].tolist()
y_pred = getRx(sdata['pe rslt'])
print(makeConfusion(y_true,y_pred,["Neg","Pos"]))
print('Accuracy: %f Precision: %f Recall: %f'%calculateMetrics(y_true, y_pred))

#     Neg  Pos
# Neg   80  750
# Pos    5  109
# Accuracy: 0.200212 Precision: 0.126892 Recall: 0.956140

################################################################
# Chapman data (N=860)-1 nan classified using PE Finder
# How many cases did PEFinger get right?
################################################################


cdata = pickle.load(open('results/chapman-pe-results.pkl','rb'))
cdata = cdata[cdata['disease_state'].isnull()==False] #one case
y_true = cdata['disease_state'].tolist()
y_pred = getRx(cdata['pe rslt'])
print(makeConfusion(y_true,y_pred,["Neg","Pos"]))
print('Accuracy: %f Precision: %f Recall: %f'%calculateMetrics(y_true, y_pred))

#     Neg  Pos
# Neg   41  525
# Pos    4  289
# Accuracy: 0.384168 Precision: 0.355037 Recall: 0.986348

# There must be a bug in my code. This is what a result looks like:

# classrslts(context_document=__________________________________________
#, exam_type='ctpa', report_text=' -***-1.  PARTIAL THROMBUS IN THE LEFT INTERNAL JUGULAR VEIN, LEFT INNOMINATE VEIN, AND LEFT SUBCLAVIAN VEIN, WITH PREFERENTIAL DRAINAGE  INTO THE SUPERIOR VENA CAVA AND AZYGOUS SYSTEM THROUGH PARASPINAL COLLATERALS AND THE INTERNAL MAMMARY VEINS. -***-2.  PROBABLE ATELECTASIS OF THE LEFT LUNG BASE. -***-3.  SMALLER LEFT PLEURAL EFFUSION. -***-CLINICAL HISTORY:  Patient is an eight-year-old female with T-cell lymphoblastic lymphoma now on induction therapy with tachycardia and tachypnea.  -***-4.  NO EVIDENCE OF PULMONARY EMBOLISM.  -***-', classification_result={'pulmonary_embolism': (2, "\n<tagObject>\n<id> 318621783950543567123196660883053936642 </id>\n<phrase> PULMONARY EMBOLISM </phrase>\n<literal> pulmonary embolism </literal>\n<category> ['pulmonary_embolism'] </category>\n<spanStart> 15 </spanStart>\n<spanStop> 33 </spanStop>\n<scopeStart> 0 </scopeStart>\n<scopeStop> 34 </scopeStop>\n</tagObject>\n", [])})

# I am looking at the key of the dictionary under classifier_result, is this correct?
