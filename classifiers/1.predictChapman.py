#!/env/python
'''
predictChapman.py will produce results for PEfinder using both Chapman 
and Stanford input data. Both are saved to results as [dataset]-pe-results.pkl.
My understanding is that this is a rule based classifier, and so we don't need
to do any kind of training, cross validation, etc, just run on the data. 
Accuracy calculations are in 2.calculateAccuracy.py

'''

import pandas
import pickle

from utils import analyze_result

# Let's make sure we are working from CODE HOME
CODE_HOME = os.environ["CODE_HOME"]
os.chdir(CODE_HOME)

############################################################################
# PE Finder on Chapman Data
############################################################################

# Read in Chapman data, and knowledge base
data = pandas.read_csv('chapman-data/chapman_df.tsv',sep="\t",index_col=0)
kb = pickle.load(open('chapman-data/chapman-kb.pkl','rb'))
data = data.dropna(subset=['impression'])


data["pe rslt"] = \
data.apply(lambda x: analyze_report(x["impression"], 
                                     kb["modifiers"], 
                                     kb["targets"],
                                     kb["rules"],
                                     kb["schema"]), axis=1)

# Results are at data['pe rslt'][0].classification_result
pickle.dump(data,open('results/chapman-pe-results.pkl','wb'))

############################################################################
# PE Finder on Stanford Data
############################################################################

# Read in Chapman data, and knowledge base
sdata = pandas.read_csv('stanford-data/stanford_df.tsv',sep="\t",index_col=0)
sdata = sdata.dropna(subset=['impression']) #N=944

sdata["pe rslt"] = \
sdata.apply(lambda x: analyze_report(x["impression"], 
                                     kb["modifiers"], 
                                     kb["targets"],
                                     kb["rules"],
                                     kb["schema"]), axis=1)

# Results are at data['pe rslt'][0].classification_result
pickle.dump(sdata,open('results/stanford-pe-results.pkl','wb'))
