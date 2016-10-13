# Predicting PE from Radiology Reports

We are asking the following questions, from [Chapman](docs/chapman_pefinder.pdf)

- Was the patient positive for a pulmonary embolism? 
- Was the pulmonary embolism acute or chronic? 
- Was there uncertainty regarding the diagnosis of PE? 
- Was the exam of diagnostic quality?

For this work, I will run a Stanford dataset through the PE Finder, and compare this to a traditional NLP method. I will also do the opposite for the original (Ohio) data used with PEFinder (i.e, run the Chapman data that has been analyzed with PE Finder through some NLP method we develop.


## Files and data
A little about the file organization:

- [.dev](.dev): Is a hidden folder with scripts that were not used for the final analysis.
- [chapman-data](chapman-data): is data from Chapman's work, for comparison of methods
- [stanford-data](stanford-data): includes data from Stanford (not currently in the repo, as it's too big)
- [Dockerfile](Dockerfile): is the specification for a Docker image, `vanessa/pe-predictive`, to run the analysis.

More detail on the analysis scripts is included below setup.


## Setting up the environment
First you will need to [install Docker](https://docs.docker.com/engine/installation/). Once installed, you can build the image as follows:

      docker build -t vanessa/pe-predictive .

This will build an image called `vanessa/pe-predictive`. We will then run the container and map the files in the present working directory to the container, so they are shared:

      docker run -it -v $PWD/:/code vanessa/pe-predictive bash

At this point, you are inside the container. The files in the folder on your machine are mapped as a volume. The working directory will be the folder `/code` and within this folder you will see the same files as on your local machine. All python dependencies are installed, and you should use `python3`, `python`, `ipython3` and `pip3`, and these have been aliased to run with their counterparts (without the 3). When you use `ipython` and need to paste, the paste magic (`%paste`) will not work, but you can accomplish the same thing with `%cpaste` followed by CONTROL+D. Finally, an environmental variable called `CODE_HOME` is set to make sure we don't have path errors in our scripts.

You can of course install on your local machine, but then you will need to install dependencies (not recommended).


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
 

## Analysis steps

### Filter and preparing data

**0.reportsPrep.py**
The script [0.reportsPrep.py](0.reportsPrep.py) simply loads the data (from what I have, the `final_3.csv`). It summarizes counts for each of the class labels, along with columns provided and shows the change in size before and after filtering. The final task is to save a filtered dataset from the raw data, for each of chapman and Stanford, in the [stanford-data](stanford-data) and [chapman-data](chapman-data) folders, each in the format that the other's classifier needs. Note: the Chapman data was produced according to the notebook [DocumentClassification.ipynb](../chapman/notebooks/DocumentClassification.ipynb) up to the line 22:

      data, kb = get_data()

and then saved:

      # 'id', 'impression', 'disease_state', 'uncertainty', 'quality', 'historicity'
      data.to_csv("chapman-data/chapman_df.tsv",sep="\t")
      
      # 'modifiers', 'schema', 'rules', 'targets'
      pickle.dump(kb,open('chapman-data/chapman-kb.pkl','wb'))


**1.predictChapman.py**
This will use both Utah (Chapman) and Stanford datasets to build the rule-based (Chapman) model, using PEpredict. 

**2.predictStanford.py**
This will use both Utah (Chapman) and Stanford datasets to build the rule-based (Chapman) model, using PEpredict. 





The script [1.countVectorizer.py](1.countVectorizer.py) (also mostly self documented) uses the scikit learn count vectorizer to build ensemble tree classifiers for each of the same holdout groups. There were much better results for this method (and I believe that I reproduced the original result) however it was very sensitive to the data used for train and test:

### Impressions

	holdout(3)-train(2|4)
	Accuracy: 0.824561403509

	holdout(2)-train(3|4)
	Accuracy: 0.915611814346

	holdout(4)-train(2|3)
	Accuracy: 0.74375


### Entire Reports


	holdout(3)-train(2|4)
	Accuracy: 0.842105263158

	holdout(2)-train(3|4)
	Accuracy: 0.909282700422

	holdout(4)-train(2|3)
	Accuracy: 0.73125


Using 2|3 to train and 4 to test has equivalent results between using the entire report and just impressions. There is some improvement in using the full report when using 3 to test, and slight worse performance for full reports when using set 2 to test. Likely if we did these many times, we would see there is some variance (and the two aren't significantly different), but I have not yet tested this.


## Long Term Goals
- Obviously, build a better machine learning classifier
- It is not adequate to start with an already pre-processed data file - we will talk about the entire pipeline from data collection through the end of analysis. We want to be able to acquire new data and feed it seamlessly into this pipeline with minimal manual pain.


## Questions I have
- Yu mentioned that the batches could be used in sets for training/testing, and not to use 1 and 2 as they are not independent of one another. My larger question is why the data should be separated into batches to begin with? In other words, why not just combine data that is not redundant (sets 2,3,4) and then do 10 fold cross validation? 
- The doc2vec can also produce mean vectors for groups of things - we could probably make a classifier to predict larger things about the report.
