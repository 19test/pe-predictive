'''
predictCNN.py will produce the results for the CNN Word model using both
Chapman and Stanford input data. Both are saved to results as
[dataset]-pe-results.pkl. The training of this dataset is done prior in
using scripts in the cnn_model directory on a hand labeled dataset
which should not have overlap with the stanford dataset - definitely
not the Chapman dataset.
'''

import pandas
import pickle
import os
import sys
sys.path.append('cnn_model/src')
from prediction import predict


# Let's make sure we are working from CODE HOME
#CODE_HOME = os.environ["CODE_HOME"]
#os.chdir(CODE_HOME)

############################################################################
# CNN Word on Chapman Data
############################################################################

# Read in Chapman data, and knowledge base
data = pandas.read_csv('chapman-data/chapman_df.tsv',sep="\t",index_col=0)
data = data.dropna(subset=['impression'])


data['pe_rslt'] = predict(data['impression'].values)

# Results are at data['pe rslt'][0].classification_result
pickle.dump(data,open('results/chapman-cnn-results.pkl','wb'))

############################################################################
# CNN Word on Stanford Data
############################################################################

# Read in Chapman data, and knowledge base
sdata = pandas.read_csv('stanford-data/stanford_df.tsv',sep="\t",index_col=0)
sdata = sdata.dropna(subset=['impression']) #N=944

sdata['pe_rslt'] = predict(sdata['impression'].values)

# Results are at data['pe rslt'][0].classification_result
pickle.dump(sdata,open('results/stanford-cnn-results.pkl','wb'))
