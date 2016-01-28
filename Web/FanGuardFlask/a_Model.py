import numpy as np
import pickle
import MySQLdb as mdb
import pandas as pd

from sklearn import cross_validation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier

import modelmaker


def GetSFModel(name,con,path,cut):

    sql_query = 'SELECT * FROM sfilterMeta WHERE name="%s"' % name
    
    query_results=pd.read_sql_query(sql_query,con)

    fmodel = "%s/%s" % (path,query_results['mf'][0])
    fv1 = "%s/%s" % (path,query_results['v1f'][0])
    fv2 = "%s/%s" % (path,query_results['v2f'][0])

    cutval = query_results[cut][0]
    
    with open(fmodel, 'rb') as fid:
        model = pickle.load(fid)
    with open(fv1, 'rb') as fid:
        v1 = pickle.load(fid)
    with open(fv2, 'rb') as fid:
        v2 = pickle.load(fid)       

    return model, v1, v2, cutval

def GetPFModel(name,con,path):

    sql_query = 'SELECT * FROM pfilterMeta WHERE name="%s"' % name
    
    query_results=pd.read_sql_query(sql_query,con)

    fmodel = "%s/%s" % (path,query_results['mp'])
    fv1 = "%s/%s" % (path,query_results['v1p'])
    fv2 = "%s/%s" % (path,query_results['v2p'])
    
    with open(fmodel, 'rb') as fid:
        model = pickle.load(fid)
    with open(fv1, 'rb') as fid:
        v1 = pickle.load(fid)
    with open(fv2, 'rb') as fid:
        v2 = pickle.load(fid)       
    return model, v1, v2

def ModelIt(name,cut,df):

    con = mdb.connect('localhost', 'rbt', 'elil85', 'InsightPaths')
    path = "/Users/ruthtoner/CodingMacros/ProjectInsight/Fanguard/files"
    
    models,vs1,vs2,cutval = GetSFModel(name,con,path,cut)
    p1 = modelmaker.model_tester(df,models,vs1,vs2)

    print "CUT IS",cutval
    c1 = (p1>cutval).astype(int)
    
    return c1
