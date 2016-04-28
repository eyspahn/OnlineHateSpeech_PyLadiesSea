# Online Hate Speech Modeling using Python and reddit comment data
Repo for presenting online hate speech project to PyLadies Seattle, 4/28/2016

Primary project repo at https://github.com/eyspahn/OnlineHateSpeech

I'm going to make the Google Slideshow public...once I'm done with it. Link TBA.

## Setup

### Python packages of particular interest:
- xgboost: ```pip install xgboost```
    - you may need to ```sudo yum install gcc``` before this works
- gensim: ```pip install gensim```

### Full list of packages you'll need:
        numpy, scipy, xgboost, gensim, matplotlib, scikit-learn, nltk, flask


## Repo Files:

### Scripts:

#### ExtractComments_pyladies.py
Pulls comments from relevant subreddits out of Kaggle SQL reddit comments file (available [here](https://www.kaggle.com/reddit/reddit-comments-may-2015)) & saves comments & labels into a pandas dataframe.
Takes ~30 minutes to run on a macbook air. (Script is not optimized.)
Generates pickled file "labeledhate_pyladies.p"

#### xgb_CV_woutput_pyladies.py
Run this to perform 5-fold cross validation, while outputting ROC curves, confusion matrices, feature importances, and AUC scores.
Takes a long time to run. I was running this on a large AWS EC2 machine.

#### buildxgbmodel_pyladies.py
Run this to build the model for future use/prediction.
Creates xgboost model object ```hatepredictor_pyladies.model``` & tf-idf object ```vect_pyladies.p```

#### runfinalmodelpreds.py
Run this to run a comment through the model from the command line & get a prediction. Based on the small subset of data we worked with here.


### Data

#### hatepredictor_pyladies.model
The saved xgboost model. Load with
```bst = xgb.Booster()
bst.load_model('../data/hatepredictor_pyladies.model')```

#### labeledhate_pyladies.p.zip
Unzip to get a pickle file which is a pandas dataframe of comments and labels.

#### vect_pyladies.p
The saved (trained) TF-IDF object.


### notebooks
```Example_Results.ipynb``` - A notebook to probe the classification & word2vec models.


### webapp_v1.zip
It's not hosted yet, but I have built a website to accept comments & return hate speech classification prediction. Unzip to get a folder containing the flask app. Run it locally by running ```python application.py``` in this ("webapp_v1") directory. Then navigate in your browser to the url shown in the terminal (for me, http://127.0.0.1:5000/). Ctrl+C in the terminal to quit the app.
