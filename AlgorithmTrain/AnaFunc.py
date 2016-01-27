from __future__ import division

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import time

from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC

import dfmaker
import modelmaker

def DrawROCandThresh(df,result_prob,title):
       
    plt.rcParams['figure.figsize'] = (8.0, 6.0)
    plt.rcParams['font.size'] =  15    
        
    y_real = df['evtclass']
    y_score = result_prob
    weights = df['w']
          
        
    #print y_real.shape
    #print y_score.shape
    fpr, tpr, thresh = metrics.roc_curve(y_real, y_score, pos_label=1,sample_weight=weights)

    got60=False
    got80=False
    got90=False
    dict_cuts={}
    
    for i,t in enumerate(thresh):

        if tpr[i] > 0.60 and t<1 and not got60:
            print "TP=60p: thresh=%f, FP=%f" % (t,fpr[i])
            got60=True
            dict_cuts['60p']=t

        if tpr[i] > 0.80 and t<1 and not got80:
            print "TP=80p: thresh=%f, FP=%f" % (t,fpr[i])
            got80=True
            dict_cuts['80p']=t
            
        if tpr[i] > 0.90 and t<1 and not got90:
            print "TP=90p: thresh=%f, FP=%f" % (t,fpr[i])
            got90=True
            dict_cuts['90p']=t
    
    roc_auc = metrics.auc(fpr, tpr,reorder=True)
    
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title("Cut Threshold versus Selection Rate: %s" % title)
    plt.legend(loc="lower right")
    plt.show()
    
    plt.figure()
    plt.plot(thresh, tpr, label='True Positive Rate')
    plt.plot(thresh, fpr, label='False Positive Rate')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Threshold')
    plt.ylabel('Rate')
    plt.title('Cut Threshold versus Selection Rate: %s' % title)
    plt.legend(loc="lower left")
    plt.show()
    
    return dict_cuts

def DrawScatter(df_Test, result,title):
    plt.rcParams['figure.figsize'] = (8.0, 6.0)
    plt.rcParams['font.size'] =  15
    
    str_words = [len(w.split()) for w in df_Test['words']]
    spoiler_index = df_Test[df_Test['evtclass'] == 1].index.tolist()
    regular_index = df_Test[df_Test['evtclass'] == 0].index.tolist()

    plt.figure()
    plt.scatter(np.asarray(str_words)[regular_index], \
                result[regular_index],marker="o",alpha=0.25,label="Regular",color="blue")
    plt.scatter(np.asarray(str_words)[spoiler_index], \
                result[spoiler_index],marker="o",alpha=0.1,label="Spoiler",color="red")
    plt.xlim([0, 200])
    plt.ylim([0, 1.05])
    plt.xlabel('Number of Words in Text')
    plt.ylabel('Spoiler Probability')
    plt.title('%s: Text Length versus Spoiler Prediction' % title)
    plt.legend(loc="lower right")

    plt.show()

def DrawBestWords(trained_model,trained_vocab, trained_tag_vocab, title):

    max_idx_v1 = len(trained_vocab.get_feature_names())
    max_idx_v2 = max_idx_v1 + len(trained_tag_vocab.get_feature_names())
    
    #Most important word features to Star Wars Random Forest:
    importances = trained_model.feature_importances_

    #Ranked from most to least important:
    indices_ = np.argsort(importances)
    indices = indices_[::-1]

    #Index and name of each feature:
    names=[]
    imp=[]

    for j,ind in enumerate(indices):
        
        #print "Looking for ind",ind
        
        if ind < max_idx_v1:
            #print "Look in v1, under", max_idx_v1
            names.append(trained_vocab.get_feature_names()[ind])
        elif ind < max_idx_v2:
            #print "Look in v2, under", max_idx_v2
            names.append(trained_tag_vocab.get_feature_names()[ind-max_idx_v1]+" (T)")   
        else:
            #print "It's length"
            names.append("WCOUNT")
        imp.append(importances[ind])
        #print j,trained_vocab.get_feature_names()[ind],importances[ind]
        if j > 20:

            break

    plt.rcParams['figure.figsize'] = (14.0, 6.0)
    plt.rcParams['font.size'] =  15

    fig = plt.figure()

    width = 0.35


    ind = np.arange(len(names))
    plt.bar(ind, imp,width)
    plt.xticks(ind + width, names,rotation='vertical',fontsize='large')
    plt.xlim(-width,len(ind)+width)

    plt.xlabel('Word',fontsize='large')
    plt.ylabel('Importance to Model',fontsize='large')
    plt.title('Most Important Words: %s Spoilers' % title,fontsize='large')

def Train_A_Model_Direct(tag, model, vocab1,vocab2,df_Train,df_Test):

    #Train the actual model:
    trained_model,trained_vocab,tagged_vocab = modelmaker.model_trainer(df_Train,model,vocab1,vocab2)

    #Get predictions
    result = modelmaker.model_tester(df_Test,trained_model,trained_vocab,tagged_vocab)

    return trained_model, trained_vocab, tagged_vocab, result, df_Test

    
def Train_A_Model(tag, model, vocab1,vocab2,myconfig,downsample=True):

    df_all = dfmaker.get_train_dfs(tag,myconfig)
    df_Train, df_Test = dfmaker.GenerateTestTrain(df_all)

    #Train the actual model:
    trained_model,trained_vocab,tagged_vocab = modelmaker.model_trainer(df_Train,model,vocab1,vocab2,downsample)

    #Get predictions
    result = modelmaker.model_tester(df_Test,trained_model,trained_vocab,tagged_vocab)

    return trained_model, trained_vocab, tagged_vocab, result, df_Test

def GetDFs(tag,myconfig):
    df_all = dfmaker.get_train_dfs(tag,myconfig)
    df_Train, df_Test = dfmaker.GenerateTestTrain(df_all)
    return df_all, df_Train, df_Test

def CV_A_Model(model, vocab1,vocab2,df,nfold=5,downsample=True):

    start_time = time.time()
    #Test with cross-validation:
    modelmaker.model_cv(df,model,vocab1,vocab2,nfold,downsample)
    print "Time elapsed:",(time.time() - start_time)
    
def CV_A_Model_FromSQL(tag, model, vocab1,vocab2,myconfig,nfold=5,downsample=True):

    df_all = dfmaker.get_train_dfs(tag,myconfig,downsample)
    df_Train, df_Test = dfmaker.GenerateTestTrain(df_all)
    
    #Test with cross-validation:
    modelmaker.model_cv(df_Train,model,vocab1,vocab2,nfold,downsample)

def CV_FrontFilter(model, nolist, vocab1,vocab2,myconfig,nfold=5,downsample=True):

    df_all = dfmaker.get_nolist_dfs(tag,nolist,myconfig)

    for itest in range(1,len(nolist)+1):
        df_Test,df_Train = dfmaker.GenerateTestTrainFront(df,itest)
        Train_A_Model_Direct(tag, model, vocab1,vocab2,df_Train,df_Test)
        
#def Test_Front_Model(tag, nolist, model, vocab1,vocab2,title,myconfig):
#
#    print "NOW TESTING:",tag
#
#    #df_ip = dfmaker.get_train_dfs(tag,myconfig)
#
#    df_nonip = dfmaker.make_front_dfs(tag,nolist,myconfig)
#    
#    df_Train, df_Test = dfmaker.GenerateTestTrain(df_all)
#
#    #Test with cross-validation:
#    modelmaker.model_cv(df_all,model,vocab1,vocab2)
#    #print "Mean AUC = %d, Std = %d" % (mean_auc,std_auc)
#
#    #Train the actual model:
#    trained_model,trained_vocab,tagged_vocab = modelmaker.model_trainer(df_Train,model,vocab1,vocab2)
#
#    result = modelmaker.model_tester(df_Test,trained_model,trained_vocab,tagged_vocab)
#    DrawROCandThresh(df_Test,result,title)
#    #DrawScatter(df_Test,result,title)
#    DrawBestWords(trained_model,trained_vocab,tagged_vocab,title)
#
#    return trained_model, trained_vocab, result, df_Test
