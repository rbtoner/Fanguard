from __future__ import division

import numpy as np
import MySQLdb as mdb
import pandas as pd
import re

from sklearn import cross_validation
from sklearn import metrics
from sklearn.utils import shuffle

import ConfigParser

def is_spoil(x):
    if 'spoil' in x['taglist'].lower():
        return False
    else:
        return True
    
def has_spoil(x):
    if 'spoil' in x['alltext'].lower():
        return False
    else:
        return True
    
def snipspo(x):
    x = re.sub("\S*[Ss][Pp][Oo][Ii][Ll]\S*","",x)
    return x
        
    
def clean_tags(x):
    tagvals=""
    for t in x.split('",'):
        t = re.sub("[^a-zA-Z ]","", t )
        t = re.sub("\S*[Ss][Pp][Oo][Ii][Ll]\S*","",t)
        tagvals += t
        
        #if "spoil" not in t:
        #    tagvals += t
            
    return tagvals

def get_train_dfs(dbtag,myconfig):

    #Grab configs:
    config = ConfigParser.RawConfigParser()
    config.read(myconfig) 

    #MySQL info:
    db_username = config.get('DB', 'username')
    db_pwd = config.get('DB', 'pwd')
    
    #DB connection:
    #con = mdb.connect('localhost', db_username, db_pwd, 'testdb')
    con = mdb.connect('localhost', db_username, db_pwd, 'InsightData')
    
    query = 'SELECT * FROM tumblr_%s_spoil' % dbtag
    df_spo = pd.read_sql(query, con)
    
    query = 'SELECT * FROM tumblr_%s_regular' % dbtag
    df_tot = pd.read_sql(query, con)
    
    df_tot = df_tot[df_tot.apply(is_spoil,axis=1)]
    df_tot = df_tot[df_tot.apply(has_spoil,axis=1)]
        
    df_tot['alltext'] = df_tot['alltext'].apply(snipspo)
    words_tot = np.asarray(df_tot['alltext'])
    class_tot = np.zeros(words_tot.shape[0])
    df_tot['taglist'] = df_tot['taglist'].apply(clean_tags)

    df_spo['alltext'] = df_spo['alltext'].apply(snipspo)
    words_spo = np.asarray(df_spo['alltext'])
    class_spo = np.ones(words_spo.shape[0])
    df_spo['taglist'] = df_spo['taglist'].apply(clean_tags)
    
    wtot = df_tot['dtime'].sum()
    wspo = df_spo['dtime'].sum()
    
    wval = wspo/wtot
    
    weight_tot = np.full(words_tot.shape[0],wval,dtype=float)
    weight_spo = np.ones(words_spo.shape[0])
    
    cols_tot = {'words':words_tot,'evtclass':class_tot,'w':weight_tot,\
                'wcount':np.asarray(df_tot['count']), \
                'id':np.asarray(df_tot['id']),  \
                'timestamp':np.asarray(df_tot['timestamp']), \
                'taglist':np.asarray(df_tot['taglist'])}
                
    df_new_tot = pd.DataFrame(cols_tot,index=range(1,words_tot.size+1))

    cols_spo = {'words':words_spo,'evtclass':class_spo,'w':weight_spo,\
               'wcount':np.asarray(df_spo['count']), \
               'id':np.asarray(df_spo['id']),\
                'timestamp':np.asarray(df_spo['timestamp']), \
                'taglist':np.asarray(df_spo['taglist'])}
                
    df_new_spo = pd.DataFrame(cols_spo,index=range(words_tot.size+1,words_tot.size+words_spo.size+1))
    
    df_new = pd.concat([df_new_tot,df_new_spo])
    
    df_new = df_new.drop_duplicates()
    
    avg_len = df_new['wcount'].sum()/df_new.shape[0]
    std_len = df_new['wcount'].std()
    
    df_new['wcount'] = (df_new['wcount'] - std_len)/avg_len
    
    return df_new

#Make a testing and training dataframe:
def GenerateTestTrain(df):
    
    df = shuffle(df,random_state=15)
    
    X_train, X_test, y_train, y_test = cross_validation.train_test_split( \
        df[['words','w','taglist','wcount']], df['evtclass'], test_size=0.4, \
        random_state=0)

    df_tr = pd.concat([X_train, y_train], axis=1, join_axes=[X_train.index])
    df_tr = df_tr.reset_index()

    df_te = pd.concat([X_test, y_test], axis=1, join_axes=[X_test.index])
    df_te = df_te.reset_index()
    
    return df_tr,df_te

def get_nolist_dfs(tag,nolist,myconfig):

    df_ip = get_train_dfs(tag,myconfig)
    df_ip['evtclass'] = np.zeros(df_ip.shape[0])
    df_tot = df_ip.drop_duplicates()
    
    for i,t in enumerate(nolist):
        df_ = get_train_dfs(t,myconfig)
        df_ = df_.drop_duplicates()
        df_['evtclass'] = np.full(df_.shape[0],i+1)
        df_tot = pd.concat([df_tot,df_],ignore_index=True)

    return df_tot

def GenerateTestTrainFront(df,itest):

    #print "Total size",df.shape
    
    df = shuffle(df,random_state=15)

    df_test_noip = df[df['evtclass']==itest]

    #print "No IP test shape",df_test_noip.shape
    
    df_train = df[df['evtclass']!=itest]

    #print "Everything except MIP:",df_train.shape
    
    df_ip = df_train[df_train['evtclass']==0]

    #print "Total IP only:",df_ip.shape
    
    df_train_noip = df_train[df_train['evtclass']!=0]

    #print "Total No IP train",df_train_noip.shape
    
    X_ip_train, X_ip_test, y_ip_train, y_ip_test = \
      cross_validation.train_test_split( \
            df_ip[['words','w','taglist','wcount']], \
            df_ip['evtclass'], test_size=0.4, \
            random_state=0)

    df_tr_ip = pd.concat([X_ip_train, y_ip_train], axis=1, join_axes=[X_ip_train.index])

    #print "Train IP type",type(df_tr_ip)
    #print "Train IP size",df_tr_ip.shape

    df_tr_ip['evtclass'] = np.ones(df_tr_ip.shape[0])
    df_train_noip['evtclass'] = np.zeros(df_train_noip.shape[0])   

    df_tr = pd.concat([df_tr_ip,df_train_noip],ignore_index=True)
    df_tr = df_tr.reset_index()
    df_tr = shuffle(df_tr,random_state=15)

    #print "Total Train type",type(df_tr)
    #print "Total Train size",df_tr.shape

    df_te_ip = pd.concat([X_ip_test, y_ip_test], axis=1, join_axes=[X_ip_test.index])

    #print "Test IP type",type(df_te_ip)
    #print "Test IP size",df_te_ip.shape

    df_te_ip['evtclass'] = np.ones(df_te_ip.shape[0])
    df_test_noip['evtclass'] = np.zeros(df_test_noip.shape[0]) 
    
    df_te = pd.concat([df_te_ip,df_test_noip],ignore_index=True)
    df_te = df_te.reset_index()
    df_te = shuffle(df_te,random_state=15)   

    #print "Total Test type",type(df_te)
    #print "Total Test size",df_te.shape
    
    return df_tr,df_te
