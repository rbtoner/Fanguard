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

#
def DrawROCandThresh(df,result_prob,title):
    """Draw ROC and Threshold plots:
    df = data frame
    result_prob = vector of predictions
    title = title to use for plot
    """
    
    #Set drawing params
    plt.rcParams['figure.figsize'] = (8.0, 6.0)
    plt.rcParams['font.size'] =  20    

    #Labels, Prediction, and weights (not used)
    y_real = df['evtclass']
    y_score = result_prob
    weights = df['w']
                  
    #print y_real.shape
    #print y_score.shape

    #Calculate FPR, TPR, and Threshold from sklearn metrics:
    fpr, tpr, thresh = metrics.roc_curve(y_real, y_score, pos_label=1,sample_weight=weights)

    #Levels to find:
    got60=False
    got80=False
    got90=False
    dict_cuts={}

    #Find the cutoffs for each level of True Positive Rate:
    for i,t in enumerate(thresh):

        #60% of spoilers caught:
        if tpr[i] > 0.60 and t<1 and not got60:
            print "TP=60p: thresh=%f, FP=%f" % (t,fpr[i])
            got60=True
            dict_cuts['60p']=t

        #80% of spoilers caught:
        if tpr[i] > 0.80 and t<1 and not got80:
            print "TP=80p: thresh=%f, FP=%f" % (t,fpr[i])
            got80=True
            dict_cuts['80p']=t

        #90% of spoilers caught:
        if tpr[i] > 0.90 and t<1 and not got90:
            print "TP=90p: thresh=%f, FP=%f" % (t,fpr[i])
            got90=True
            dict_cuts['90p']=t

    #Make the ROC curve:
    roc_auc = metrics.auc(fpr, tpr,reorder=True)

    #Plot the ROC:
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title("ROC Curve: %s" % title)
    plt.legend(loc="lower right")
    plt.show()

    #Plot the threshold:
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

    #Return the cuts for insertion into MySQL:
    return dict_cuts


def DrawScatter(df_Test, result,title):
    """Draw Scatter of Prediction vs Length:
    df_Ttest = test data frame
    result = vector of predictions
    title = title to use for plot
    """
       
    plt.rcParams['figure.figsize'] = (8.0, 6.0)
    plt.rcParams['font.size'] =  20

    #Get the total number of words:
    str_words = [len(w.split()) for w in df_Test['words']]
    spoiler_index = df_Test[df_Test['evtclass'] == 1].index.tolist()
    regular_index = df_Test[df_Test['evtclass'] == 0].index.tolist()

    #Draw the scatter plot:
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
    """Draw importance of features (with words):
    trained_model = the Random Forest (trained0
    trained_vocab = Vocab (Trained) for body words
    trained_tag_vocab = Vocab (Trained) for tag words
    title = title to use for plot
    """
    #Get first index of features for both vocabularies:
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

    #Loop over indices:
    for j,ind in enumerate(indices):
        
        #print "Looking for ind",ind

        #Body words:
        if ind < max_idx_v1:
            #print "Look in v1, under", max_idx_v1
            names.append(trained_vocab.get_feature_names()[ind])
        elif ind < max_idx_v2: #Tag Words
            #print "Look in v2, under", max_idx_v2
            names.append(trained_tag_vocab.get_feature_names()[ind-max_idx_v1]+" (T)")   
        else: #It's the word count:
            #print "It's length"
            names.append("WCOUNT")
        imp.append(importances[ind])
        #print j,trained_vocab.get_feature_names()[ind],importances[ind]
        if j > 20: #Only show 20 words

            break

    #Print the plot:
    plt.rcParams['figure.figsize'] = (14.0, 6.0)
    plt.rcParams['font.size'] =  20

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
    """Given a testing and training dataframe, train a model
    tag = tag name
    model = model object (sklearn) to train to
    vocab1 = Body vocab
    vocab2 = Tag vocab
    df_Train = Train dataframe
    df_Test = Test dataframe
    """
    
    #Train the actual model:
    trained_model,trained_vocab,tagged_vocab = modelmaker.model_trainer(df_Train,model,vocab1,vocab2)

    #Get predictions
    result = modelmaker.model_tester(df_Test,trained_model,trained_vocab,tagged_vocab)

    #Return trained model, vocab, prediction:
    return trained_model, trained_vocab, tagged_vocab, result, df_Test

    
def Train_A_Model(tag, model, vocab1,vocab2,myconfig,downsample=True):
    """Given a tag name, train a model
    tag = tag name
    model = model object (sklearn) to train to
    vocab1 = Body vocab
    vocab2 = Tag vocab
    myconfig = config file for accessing MySQL database
    downsample = force spoiler and non-spoiler sets to have same size
    """

    #Get the data and make Test/Train frames:
    df_all = dfmaker.get_train_dfs(tag,myconfig)
    df_Train, df_Test = dfmaker.GenerateTestTrain(df_all)

    #Train the actual model:
    trained_model,trained_vocab,tagged_vocab = modelmaker.model_trainer(df_Train,model,vocab1,vocab2,downsample)

    #Get predictions
    result = modelmaker.model_tester(df_Test,trained_model,trained_vocab,tagged_vocab)

    #Return trained model, vocab, prediction:
    return trained_model, trained_vocab, tagged_vocab, result, df_Test

def Train_Final_Model(tag, model, vocab1,vocab2,myconfig,downsample=True):
    """Given a tag name, train a model WITH CROSS-VALIDATION
    tag = tag name
    model = model object (sklearn) to train to
    vocab1 = Body vocab
    vocab2 = Tag vocab
    myconfig = config file for accessing MySQL database
    downsample = force spoiler and non-spoiler sets to have same size
    """
    df_all = dfmaker.get_train_dfs(tag,myconfig)

    #Check that CV looks fine:
    print "Checking CV (nfold=2):"
    modelmaker.model_cv(df_all,model,vocab1,vocab2,2,downsample)

    #Train the actual model:
    print "Training final model:"
    trained_model,trained_vocab1,trained_vocab2 = modelmaker.model_trainer(df_all,model,vocab1,vocab2,downsample)

    return trained_model, trained_vocab1, trained_vocab2


def GetDFs(tag,myconfig):
    """Simply return Test and Train Dataframes for a given tag
    tag = tag name
    myconfig = config file for accessing MySQL database  
    """
    #Get all data:
    df_all = dfmaker.get_train_dfs(tag,myconfig)
    #Break into test and train:
    df_Train, df_Test = dfmaker.GenerateTestTrain(df_all)

    return df_all, df_Train, df_Test

def CV_A_Model(model, vocab1,vocab2,df,nfold=5,downsample=True):
    """K-fold CV a model for time and performance
    model = model object (sklearn) to train to
    vocab1 = Body vocab
    vocab2 = Tag vocab
    df = Dataframe to sample from
    nfold = number of folds for CV
    downsample = force spoiler and non-spoiler sets to have same size
    """    

    #Start time:
    start_time = time.time()

    #Test with cross-validation:
    modelmaker.model_cv(df,model,vocab1,vocab2,nfold,downsample)

    #How much time elapsed?
    print "Time elapsed:",(time.time() - start_time)
    
def CV_A_Model_FromSQL(tag, model, vocab1,vocab2,myconfig,nfold=5,downsample=True):
    """K-fold CV a model for time and performance, calling from MySQL
    tag = tag name
    model = model object (sklearn) to train to
    vocab1 = Body vocab
    vocab2 = Tag vocab
    myconfig = config file for accessing MySQL database  
    nfold = number of folds for CV
    downsample = force spoiler and non-spoiler sets to have same size
    """    
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
