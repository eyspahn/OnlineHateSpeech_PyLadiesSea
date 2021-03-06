# -*- coding: utf-8 -*-
'''Build tf-idf + xgboost classification model using May 2015 data'''

from sklearn.feature_extraction.text import TfidfVectorizer

import xgboost as xgb
import cPickle as pickle

from string import punctuation
from nltk import word_tokenize
from nltk.stem import snowball

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')

stemmer = snowball.SnowballStemmer("english")

def load_data(filename='../data/labeledhate_pyladies.p'):
    '''
    Load data into a data frame for use in running model
    '''
    return pickle.load(open(filename, 'rb'))


def stem_tokens(tokens, stemmer):
    '''Stem the tokens.'''
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed


def tokenize(text):
    '''Tokenize & stem. Stems automatically for now.
    Leaving "stemmer" out of function call, so it works with TfidfVectorizer'''
    tokens = word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems


def main():
    classes = ['NotHate', 'SizeHate', 'GenderHate', 'RaceHate', 'ReligionHate']
    print('Loading Data')
    df = load_data()
    X = df.body
    # strip puctuation from X...should not be needed, since NLTK tokenizer should take care of this for us.
    X = X.apply(lambda x: ''.join([l for l in x if l not in punctuation]))
    y = df.label

    # relabel y labels to reflect integers [0-4] for xgboost
    for ind in range(len(classes)):
        y[(y == classes[ind])] = int(ind)

    print("Vectorizing")
    vect = TfidfVectorizer(stop_words='english', decode_error='ignore',
                           tokenizer=tokenize)

    # fit & transform comments matrix
    tfidf_X = vect.fit_transform(X)
    # Save out vect & tfidf_X
    pickle.dump(vect, open('../data/vect_pyladies.p', 'wb'))

    # develop data to train model
    xg_train = xgb.DMatrix(tfidf_X, label=y)
    print('Classifying')
    # Set up xboost parameters
    # use softmax multi-class classification to return probabilities
    param = {'objective': 'multi:softprob',
             'eta': 0.9,
             'max_depth': 6,
             'num_class': 5
             }
    watchlist = [(xg_train, 'train')]
    num_round = 120  # Number of rounds determined after running cross validation
    bst = xgb.train(param, xg_train, num_round, watchlist)

    # save model
    bst.save_model('../data/hatepredictor_pyladies.model')


if __name__ == '__main__':
    main()
