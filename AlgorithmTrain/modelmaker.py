from __future__ import division

from pandas import *
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import MySQLdb as mdb
import pandas as pd
import re

from matplotlib.ticker import NullFormatter

from sklearn import cross_validation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC

from sklearn.utils import shuffle

from sklearn import metrics

import dfmaker

def retrieve_vocab(v1,v2,df_test,downsample=False):
    """Turn df_test taglist and wordlist into set of features
    v1 = trained body vocab
    v2 = trained tag vocab
    df_test = test dataframe
    downsample = make regular and spoiler set same size
    """
    
    #Word list::
    corpus_train1 = df_test['words']
    x1 = v1.transform(corpus_train1).toarray()

    #Tag list:
    corpus_train2 = df_test['taglist']
    x2 = v2.transform(corpus_train2).toarray()

    #labels:
    y = df_test['evtclass']
    
    return x1,x2,y

def train_vocab(v1,v2,df,downsample=False):
    """Train vocab for body and tags.
    v1 = untrained body vocab
    v2 = untrained tag vocab
    df_test =  dataframe
    downsample = make regular and spoiler set same size
    """

    #Downsample if requested:
    if downsample:
        df_class1 = df[df["evtclass"]==1]
        df_class0 = df[df["evtclass"]==0]
        size0 = df_class1.shape[0]
        df_class0 = df_class0.sample(size0)
        df = pd.concat([df_class1,df_class0],ignore_index=True)
        df = shuffle(df,random_state=20)
    
    #Word list::
    corpus_train1 = df['words']
    x1 = v1.fit_transform(corpus_train1).toarray()
    
    #Tag list:
    corpus_train2 = df['taglist']
    x2 = v2.fit_transform(corpus_train2).toarray()

    #Labels:
    y = df['evtclass']
    
    return x1,x2,y,v1,v2

def make_features(v1,v2,df,downsample=False):
    """Train vocab and make vector of features
    v1 = untrained body vocab
    v2 = untrained tag vocab
    df =  dataframe
    downsample = make regular and spoiler set same size
    """

    #Downsample if requested:
    if downsample:
        df_class1 = df[df["evtclass"]==1]
        df_class0 = df[df["evtclass"]==0]
        size0 = df_class1.shape[0]
        df_class0 = df_class0.sample(size0)
        df = pd.concat([df_class1,df_class0],ignore_index=True)
        df = shuffle(df,random_state=20)
    
    #Word list::
    corpus_train1 = df['words']
    x1 = v1.fit_transform(corpus_train1).toarray()
    
    #Tag list:
    corpus_train2 = df['taglist']
    x2 = v2.fit_transform(corpus_train2).toarray()

    #Word count:    
    f_wcount = df['wcount'].values
    fl = f_wcount.reshape(len(f_wcount),1)

    #Stack them together:
    x = np.hstack([x1,x2,fl])  
    y = df['evtclass']
    
    return x,y,v1,v2

def model_trainer(df_train,model,v1,v2,downsample=False):
    """Train a model from df_train (along w/ vocabs)
    v1 = untrained body vocab
    v2 = untrained tag vocab
    model = untrained model
    df_train = train dataframe
    downsample = make regular and spoiler set same size
    """
    #Features:
    x, y, v1, v2 = make_features(v1,v2,df_train,downsample)

    #Fit the model:    
    model = model.fit(x, y)

    return model,v1,v2

def model_cv(df,model,v1,v2,n_folds=5,downsample=False):
    """Train a model from df_train (along w/ vocabs) and CV
    v1 = untrained body vocab
    v2 = untrained tag vocab
    model = untrained model
    df = train dataframe
    n_folds = number of CV folds
    downsample = make regular and spoiler set same size
    """

    #Make features:
    x,y,v1,v2 = make_features(v1,v2,df,downsample) 

    #Cross validation folds:
    kf_total = cross_validation.KFold(len(x), n_folds, shuffle=True, random_state=4)
    
    #Cross-validate!
    cvs = cross_validation.cross_val_score(model, x, y, cv=kf_total, n_jobs = 1,scoring="roc_auc")
    
    print "Mean AUC = %f, Std = %f" % (cvs.mean(),cvs.std())


def model_tester(df_test,model,v1,v2):
    """Get predictions for a model
    v1 = trained body vocab
    v2 = trained tag vocab
    model = trained model
    df_test = test dataframe
    """
    
    #Word list::
    corpus_train1 = df_test['words']
    x1 = v1.transform(corpus_train1).toarray()

    #Tag list:
    corpus_train2 = df_test['taglist']
    x2 = v2.transform(corpus_train2).toarray()
    
    #Text length:
    f_wcount = df_test['wcount'].values
    fl = f_wcount.reshape(len(f_wcount),1)

    #Features vector
    x = np.hstack([x1,x2,fl])  

    #Result:
    result_p = (model.predict_proba(x))[:,1]

    return result_p

