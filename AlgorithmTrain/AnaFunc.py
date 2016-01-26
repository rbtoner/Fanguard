from __future__ import division

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from sklearn import metrics

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
    
    #return tpr, fpr, thresholds

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


def Test_A_Model(tag, model, vocab1,vocab2,title,myconfig):

    print "NOW TESTING:",tag

    df_all = dfmaker.get_train_dfs(tag,myconfig)
    df_Train, df_Test = dfmaker.GenerateTestTrain(df_all)

    #Test with cross-validation:
    modelmaker.model_cv(df_all,model,vocab1,vocab2)
    #print "Mean AUC = %d, Std = %d" % (mean_auc,std_auc)

    #Train the actual model:
    trained_model,trained_vocab,tagged_vocab = modelmaker.model_trainer(df_Train,model,vocab1,vocab2)

    result = modelmaker.model_tester(df_Test,trained_model,trained_vocab,tagged_vocab)
    DrawROCandThresh(df_Test,result,title)
    #DrawScatter(df_Test,result,title)
    DrawBestWords(trained_model,trained_vocab,tagged_vocab,title)

    return trained_model, trained_vocab, result, df_Test

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
