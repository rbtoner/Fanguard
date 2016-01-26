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

def make_features(v1,v2,df):

    #Word list::
    corpus_train1 = df['words']
    x1 = v1.fit_transform(corpus_train1).toarray()
    
    #Tag list:
    corpus_train2 = df['taglist']
    x2 = v2.fit_transform(corpus_train2).toarray()
    
    f_wcount = df['wcount'].values
    fl = f_wcount.reshape(len(f_wcount),1)
    
    x = np.hstack([x1,x2,fl])  

    return x

def model_trainer(df_train,model,v1,v2):

    #Features:
    x = make_features(v1,v2,df_train)

    #Labels:
    y = df_train['evtclass']

    #Fit the model:    
    model = model.fit(x, y)

    return model,v1,v2

def model_cv(df,model,v1,v2,n_folds=5):
    
    x = make_features(v1,v2,df) 
    
    y = df['evtclass']
    
    #kf_total = cross_validation.KFold(len(x), n_folds=10, indices=True, shuffle=True, random_state=4)
    kf_total = cross_validation.KFold(len(x), n_folds, indices=False, shuffle=True, random_state=4)
    
    #for train, test in kf_total:
    #    print train, '\n', test, '\n\n'
    #    print len(train), len(test)
        
    cvs = cross_validation.cross_val_score(model, x, y, cv=kf_total, n_jobs = 1,scoring="roc_auc")

    print cvs
    
    print "Mean AUC = %f, Std = %f" % (cvs.mean(),cvs.std())


def model_tester(df_test,model,v1,v2):

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

