# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 12:16:58 2023

@author: ZZG
"""



# from sklearnex import patch_sklearn, config_context, unpatch_sklearn
# patch_sklearn('LogisticRegression')
# unpatch_sklearn()
import os
import shap
import time
import joblib
import random
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
from itertools import chain, combinations
from xgboost import XGBClassifier
# from thundersvm import SVC
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression, ElasticNet, Lasso, Ridge, LogisticRegression, PassiveAggressiveClassifier, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier as DNN
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc, average_precision_score, confusion_matrix, brier_score_loss  ###计算roc和auc
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
# from skbio.stats.composition import multiplicative_replacement
from sklearn.model_selection import StratifiedShuffleSplit
# from tune_sklearn import TuneGridSearchCV
# from sklearn.experimental import enable_halving_search_cv
# from sklearn.model_selection import HalvingGridSearchCV
from sklearn2pmml import sklearn2pmml
from sklearn2pmml.pipeline import PMMLPipeline
import scipy.stats as st



warnings.filterwarnings('ignore')


plt.rcParams['font.size'] = 18
plt.rcParams['font.family'] = 'arial'


def custom_classification_report(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    acc = (tp+tn)/(tp+tn+fp+fn)
    sen = (tp)/(tp+fn)
    sp = (tn)/(tn+fp)
    ppv = (tp)/(tp+fp)
    npv = (tn)/(tn+fn)
    f1 = 2*(tp)/(2*(tp)+fn+fp)
    fpr = (fp)/(fp+tn)
    tpr = (tp)/(tp+fn)
    return {'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
                'Accuracy': round(acc, 3),
                'Sensitivity/Recall': round(sen, 3),
                'Specificity': round(sp, 3),
                'PPV/Precision': round(ppv, 3),
                'NPV': round(npv, 3),
                'F1-score': round(f1, 3),
                'False positive rate': round(fpr, 3),
                'True positive rate': round(tpr, 3)
            }




class DelongTest():
    def __init__(self,preds1,preds2,label,threshold=0.05):
        '''
        preds1:the output of model1
        preds2:the output of model2
        label :the actual label
        '''
        self._preds1=preds1
        self._preds2=preds2
        self._label=label
        self.threshold=threshold
        self._show_result()

    def _auc(self,X, Y)->float:
        return 1/(len(X)*len(Y)) * sum([self._kernel(x, y) for x in X for y in Y])

    def _kernel(self,X, Y)->float:
        '''
        Mann-Whitney statistic
        '''
        return .5 if Y==X else int(Y < X)

    def _structural_components(self,X, Y)->list:
        V10 = [1/len(Y) * sum([self._kernel(x, y) for y in Y]) for x in X]
        V01 = [1/len(X) * sum([self._kernel(x, y) for x in X]) for y in Y]
        return V10, V01

    def _get_S_entry(self,V_A, V_B, auc_A, auc_B)->float:
        return 1/(len(V_A)-1) * sum([(a-auc_A)*(b-auc_B) for a,b in zip(V_A, V_B)])
    
    def _z_score(self,var_A, var_B, covar_AB, auc_A, auc_B):
        return (auc_A - auc_B)/((var_A + var_B - 2*covar_AB )**(.5)+ 1e-8)

    def _group_preds_by_label(self,preds, actual)->list:
        X = [p for (p, a) in zip(preds, actual) if a]
        Y = [p for (p, a) in zip(preds, actual) if not a]
        return X, Y

    def _compute_z_p(self):
        X_A, Y_A = self._group_preds_by_label(self._preds1, self._label)
        X_B, Y_B = self._group_preds_by_label(self._preds2, self._label)

        V_A10, V_A01 = self._structural_components(X_A, Y_A)
        V_B10, V_B01 = self._structural_components(X_B, Y_B)

        auc_A = self._auc(X_A, Y_A)
        auc_B = self._auc(X_B, Y_B)

        # Compute entries of covariance matrix S (covar_AB = covar_BA)
        var_A = (self._get_S_entry(V_A10, V_A10, auc_A, auc_A) * 1/len(V_A10)+ self._get_S_entry(V_A01, V_A01, auc_A, auc_A) * 1/len(V_A01))
        var_B = (self._get_S_entry(V_B10, V_B10, auc_B, auc_B) * 1/len(V_B10)+ self._get_S_entry(V_B01, V_B01, auc_B, auc_B) * 1/len(V_B01))
        covar_AB = (self._get_S_entry(V_A10, V_B10, auc_A, auc_B) * 1/len(V_A10)+ self._get_S_entry(V_A01, V_B01, auc_A, auc_B) * 1/len(V_A01))

        # Two tailed test
        z = self._z_score(var_A, var_B, covar_AB, auc_A, auc_B)
        p = st.norm.sf(abs(z))*2

        return z,p

    def _show_result(self):
        z,p=self._compute_z_p()
 

        return z, p


def auc_plot(thersholds_train, fpr_train, tpr_train, ci_, y_train_new, y, auc_train, color='red', ls='--', label='LG', legend=True, auc_plot=True, ax=None, ctr_line=True):
    
    output_test = []
    for i, value in enumerate(thersholds_train):
        print("%f %f %f" % (fpr_train[i], tpr_train[i], value))
        
        y_predict_ = [1 if i >= value else 0 for i in y_train_new[:,1]]
        report = custom_classification_report(y, y_predict_)
        
        output_test.append([label, fpr_train[i], tpr_train[i], value, \
                        report['Sensitivity/Recall'], report['Specificity'], \
                        report['PPV/Precision'], report['NPV'],\
                        report['F1-score'], (report['Sensitivity/Recall']+report['Specificity'])/2])
    output_test_pd = pd.DataFrame(output_test)
    output_test_pd.columns = ['Dataset', 'fpr', 'tpr', 'threshold', 'Sensitivity/Recall', 'Specificity', 'PPV/Precision', 'NPV', 'F1-score', 'SS']
    output_test_pd = output_test_pd.fillna(1)

    
    if auc_plot:
        ax.plot(output_test_pd['fpr'], output_test_pd['tpr'], color=color, linestyle=ls, label=label+' (AUC = {:.2f} ± {:.2f})'.format(auc_train, ci_), lw=2)
        tpr_train_upper = np.minimum(output_test_pd['tpr'] + ci_, 1)
        tpr_train_lower = np.maximum(output_test_pd['tpr'] - ci_, 0)
        

        df_ci1_ = pd.concat([output_test_pd['fpr'], tpr_train_lower, tpr_train_upper],axis=1)
        df_ci1_.columns = ['fpr', 'lower', 'upper']
        for i in df_ci1_.index:
            ax.fill_between(df_ci1_.iloc[i:i+2]['fpr'],  df_ci1_.iloc[i:i+2]['lower'], df_ci1_.iloc[i:i+2]['upper'],  color=color, alpha=.1, edgecolor=None)                
        

        fpr_train_upper = np.minimum(output_test_pd['fpr'] + ci_, 1)
        fpr_train_lower = np.maximum(output_test_pd['fpr'] - ci_, 0)
        df_ci2_ = pd.concat([output_test_pd['tpr'], fpr_train_lower, fpr_train_upper],axis=1)
        df_ci2_.columns = ['tpr', 'lower', 'upper']
        for i in df_ci2_.index:
            ax.fill_betweenx(df_ci2_.iloc[i:i+2]['tpr'],  df_ci2_.iloc[i:i+2]['lower'], df_ci2_.iloc[i:i+2]['upper'],  color=color, alpha=.1, interpolate=False,edgecolor=None)        
        if ctr_line:
            plt.plot([0, 1], [0, 1], color='k', lw=2, linestyle='--') 
        plt.xlabel('1 - Specificity')
        plt.ylabel('Sensitivity')  
        
    else:

        ax.plot(output_test_pd['Sensitivity/Recall'], output_test_pd['PPV/Precision'], color=color, linestyle=ls, label=label+' (AUC = {:.2f} ± {:.2f})'.format(auc_train, ci_), lw=2)

    
        tpr_train_upper = np.minimum(output_test_pd['PPV/Precision'] + ci_, 1)
        tpr_train_lower = np.maximum(output_test_pd['PPV/Precision'] - ci_, 0)
        
        df_ci1_ = pd.concat([output_test_pd['Sensitivity/Recall'], tpr_train_lower, tpr_train_upper],axis=1)
        df_ci1_.columns = ['Recall', 'lower', 'upper']
        for i in df_ci1_.index:
            ax.fill_between(df_ci1_.iloc[i:i+2]['Recall'],  df_ci1_.iloc[i:i+2]['lower'], df_ci1_.iloc[i:i+2]['upper'],  color=color, alpha=.1, edgecolor=None)        

        fpr_train_upper = np.minimum(output_test_pd['Sensitivity/Recall'] + ci_, 1)
        fpr_train_lower = np.maximum(output_test_pd['Sensitivity/Recall'] - ci_, 0)
    
        df_ci2_ = pd.concat([output_test_pd['PPV/Precision'], fpr_train_lower, fpr_train_upper],axis=1)
        df_ci2_.columns = ['Precision', 'lower', 'upper']
        for i in df_ci2_.index:
            ax.fill_betweenx(df_ci2_.iloc[i:i+2]['Precision'],  df_ci2_.iloc[i:i+2]['lower'], df_ci2_.iloc[i:i+2]['upper'],  color=color, alpha=.1, interpolate=False,edgecolor=None)

        
        if ctr_line:
            class_balance = sum(y)/len(y)
            plt.plot([0, 1], [class_balance, class_balance], color='k', lw=2, linestyle='--')

        plt.xlabel('Recall')
        plt.ylabel('Precision')           
    
    plt.xlim([-0.05, 1.05])  
    plt.ylim([-0.05, 1.05])

    if legend:
        plt.legend(loc="lower right", frameon=False)
    plt.subplots_adjust(left=0.16, right=0.99, top=0.955, bottom=0.14)
    
    return output_test_pd




def compute_SD(AUC, PTB, FTB):

    Q1 = AUC/(2-AUC)
    Q2 = 2*AUC*AUC/(1+AUC)
	
    return np.sqrt((AUC*(1-AUC)+(PTB-1)*(Q1-AUC*AUC) + (FTB-1)*(Q2-AUC*AUC))/(PTB*FTB))


#%% Fig. 3C, D and Fig. 4A-F
proteomic_data_path = r'data'

MCH = pd.read_csv(os.path.join(proteomic_data_path, r'proteome.txt'), sep='\t')
MCH['group'] = MCH['Category'].str.split('-').str[1]


Train = MCH[MCH['Dataset'] == 'Train']
Internal = MCH[MCH['Dataset'] == 'Internal']
MCH2 = MCH[MCH['Dataset'] == 'External1']
MCH3 = MCH[MCH['Dataset'] == 'External2']
MCH4 = MCH[MCH['Dataset'] == 'External3']
MCH5 = MCH[MCH['Dataset'] == 'External4']

enc = preprocessing.OrdinalEncoder()

Train_y = list(chain.from_iterable(enc.fit_transform(Train[['group']].values)))
Internal_y = list(chain.from_iterable(enc.fit_transform(Train[['group']].values)))
MCH2_y = list(chain.from_iterable(enc.fit_transform(MCH2[['group']].values)))
MCH3_y = list(chain.from_iterable(enc.fit_transform(MCH3[['group']].values)))
MCH4_y = list(chain.from_iterable(enc.fit_transform(MCH4[['group']].values)))
MCH5_y = list(chain.from_iterable(enc.fit_transform(MCH5[['group']].values)))


combination_list = [[('AMBP', 'FN1', 'LUM', 'TIMP1', 'B2M')]]




time_count = 2

combine_set_all = []
count = 0
output_auc = []
for each_com in combination_list:

    x_train_pd = Train[['SampleID'] + list(each_com[0])]
    x_train_pd = x_train_pd.set_index('SampleID')
    x_train_pd = x_train_pd.div(x_train_pd.mean(axis=0), axis=1)
    y_train_pd = Train[['SampleID', 'group']]
    y_train_pd = y_train_pd.set_index('SampleID')
    

    x_internal_pd = Internal[['SampleID'] + list(each_com[0])]
    x_internal_pd = x_internal_pd.set_index('SampleID')
    x_internal_pd = x_internal_pd.div(x_internal_pd.mean(axis=0), axis=1)
    y_internal_pd = Internal[['SampleID', 'group']]
    y_internal_pd = y_internal_pd.set_index('SampleID')
    

    start = time.time()
    #########################################################################################

    ##LR
    # space = {'penalty':[ 'l2'],  'C': np.logspace(-3, 3, 6),  "solver":['lbfgs',  'saga']}
    space = {'penalty':['l1', 'l2', 'elasticnet'],  'C': np.logspace(-3, 3, 6),  "solver":['lbfgs',  'saga']}
    model_to_tune = LogisticRegression()
    grid_search=GridSearchCV(model_to_tune, space, scoring="roc_auc", cv=5, n_jobs=-1,  verbose = 1)

    
    #XGB
    # space = {'n_estimators':list(np.arange(1,1000,50))}
    # model_to_tune = XGBClassifier()
    # grid_search = None
    # grid_search=GridSearchCV(model_to_tune, space, scoring="roc_auc", cv=inner_cv, n_jobs=n_jobs, verbose = 0)
                
    # #NW
    # space = {'alpha': 10.0 ** -np.arange(1, 5), \
    #               'hidden_layer_sizes':np.arange(2, 5), \
    #               'activation':['identity', 'logistic', 'tanh', 'relu'], \
    #               'solver':['lbfgs','sgd', 'adam'], \
    #               'learning_rate':['constant', 'invscaling', 'adaptive']}
    # model_to_tune = DNN(max_iter=500)
    # grid_search = None
    # grid_search=GridSearchCV(model_to_tune, space, scoring="roc_auc", cv=inner_cv, n_jobs=n_jobs, verbose = 0)        


    #SGD
    # space = {'loss':['hinge', 'log','modified_huber','squared_hinge','perceptron', 'huber','epsilon_insensitive','squared_epsilon_insensitive'], \
    #               'penalty':['l2','l1','elasticnet'], \
    #               'alpha':[0.0001,0.0002,0.0003,0.0004,0.00005,0.00001], \
    #               'fit_intercept':[False]}

    # model_to_tune = SGDClassifier()
    # grid_search = None
    # grid_search=GridSearchCV(model_to_tune, space, scoring="roc_auc", \
    #                   cv=inner_cv, n_jobs=n_jobs, verbose = 0)

    # RF
    # space = {'n_estimators': list(np.arange(20, 220, 20)), \
    #               'max_depth': list(np.arange(2, 10)), \
    #               'criterion': ['gini','entropy']}

    # model_to_tune = RandomForestClassifier()
    # grid_search = None
    # grid_search=GridSearchCV(model_to_tune, space, scoring="roc_auc", cv=inner_cv, n_jobs=n_jobs, verbose = 0)

    #########################################################################################

    
    # with config_context(target_offload="gpu:0"):
    grid_search.fit(x_train_pd.values, enc.fit_transform(y_train_pd))
    # span = time.time() - start
    
    train_best_model = grid_search.best_estimator_
    
    coef_df = pd.DataFrame({'Feature': x_train_pd.columns, 'Coefficient': train_best_model.coef_[0]})
    
    y_train_pre = train_best_model.predict_proba(x_train_pd.values)
    fpr_train, tpr_train, thersholds_train =roc_curve(enc.fit_transform(y_train_pd), y_train_pre[:,1], drop_intermediate=False)
    precision_train_pr, recall_train_pr, thresholds_train_pr = precision_recall_curve(enc.fit_transform(y_train_pd), y_train_pre[:,1])
    auc_train = auc(fpr_train,tpr_train)
    pr_train = auc(recall_train_pr,precision_train_pr)


    y_internal_pre = train_best_model.predict_proba(x_internal_pd.values)
    fpr_test, tpr_test, thersholds_test =roc_curve(enc.fit_transform(y_internal_pd), y_internal_pre[:,1], drop_intermediate=False)
    precision_test_pr, recall_test_pr, thresholds_test_pr = precision_recall_curve(enc.fit_transform(y_internal_pd), y_internal_pre[:,1])
    auc_test = auc(fpr_test,tpr_test)
    pr_test = auc(recall_test_pr, precision_test_pr)
    

    # joblib.dump(train_best_model, os.path.join(data_path, r'LR_raw.pkl'))
    # sklearn2pmml(train_best_model, os.path.join(data_path, r'LR_five_raw.pmml'), with_repr=True)


    # external1
    x_external1_pd = MCH2[['SampleID'] + list(each_com[0])]
    x_external1_pd = x_external1_pd.set_index('SampleID')
    x_external1_pd = x_external1_pd.div(x_external1_pd.mean(axis=0), axis=1)
    y_external1_pd = MCH2[['SampleID', 'group']]
    y_external1_pd = y_external1_pd.set_index('SampleID')


    y_external1_pre = train_best_model.predict_proba(x_external1_pd.values)
    fpr_external1, tpr_external1, thersholds_external1  =roc_curve(enc.fit_transform(y_external1_pd), y_external1_pre[:,1], drop_intermediate=False)
    auc_external1 = auc(fpr_external1, tpr_external1)
    
    precision_external1, recall_external1, thresholds_external1_pr = precision_recall_curve(enc.fit_transform(y_external1_pd), y_external1_pre[:,1])
    pr_external1 = auc(recall_external1, precision_external1)



    # external2
    x_external2_pd = MCH3[['SampleID'] + list(each_com[0])]
    x_external2_pd = x_external2_pd.set_index('SampleID')
    x_external2_pd = x_external2_pd.div(x_external2_pd.mean(axis=0), axis=1)
    y_external2_pd = MCH3[['SampleID', 'group']]
    y_external2_pd = y_external2_pd.set_index('SampleID')


    y_external2_pre = train_best_model.predict_proba(x_external2_pd.values)
    fpr_external2, tpr_external2, thersholds_external2  =roc_curve(enc.fit_transform(y_external2_pd), y_external2_pre[:,1], drop_intermediate=False)
    auc_external2 = auc(fpr_external2, tpr_external2)
    
    precision_external2, recall_external2, thresholds_external2_pr = precision_recall_curve(enc.fit_transform(y_external2_pd), y_external2_pre[:,1])
    pr_external2 = auc(recall_external2, precision_external2)




    # external3
    x_external3_pd = MCH4[['SampleID'] + list(each_com[0])]
    x_external3_pd = x_external3_pd.set_index('SampleID')
    x_external3_pd = x_external3_pd.div(x_external3_pd.mean(axis=0), axis=1)
    y_external3_pd = MCH4[['SampleID', 'group']]
    y_external3_pd = y_external3_pd.set_index('SampleID')


    y_external3_pre = train_best_model.predict_proba(x_external3_pd.values)
    fpr_external3, tpr_external3, thersholds_external3  =roc_curve(enc.fit_transform(y_external3_pd), y_external3_pre[:,1], drop_intermediate=False)
    auc_external3 = auc(fpr_external3, tpr_external3)
    
    precision_external3, recall_external3, thresholds_external3_pr = precision_recall_curve(enc.fit_transform(y_external3_pd), y_external3_pre[:,1])
    pr_external3 = auc(recall_external3, precision_external3)


    # external4
    x_external4_pd = MCH5[['SampleID'] + list(each_com[0])]
    x_external4_pd = x_external4_pd.set_index('SampleID')
    x_external4_pd = x_external4_pd.div(x_external4_pd.mean(axis=0), axis=1)
    y_external4_pd = MCH5[['SampleID', 'group']]
    y_external4_pd = y_external4_pd.set_index('SampleID')


    y_external4_pre = train_best_model.predict_proba(x_external4_pd.values)
    fpr_external4, tpr_external4, thersholds_external4  =roc_curve(enc.fit_transform(y_external4_pd), y_external4_pre[:,1], drop_intermediate=False)
    auc_external4 = auc(fpr_external4, tpr_external4)
    
    precision_external4, recall_external4, thresholds_external4_pr = precision_recall_curve(enc.fit_transform(y_external4_pd), y_external4_pre[:,1])
    pr_external4 = auc(recall_external4, precision_external4)
    
    # plt.plot(recall_test_zh_dg_pr, precision_test_zh_dg_pr, color='red', linestyle='--', label='LG (AUC = {0:.3f})'.format(pr_test_zh_dg), lw=2)

explainer = shap.LinearExplainer( train_best_model, x_train_pd)
shap_values = explainer(x_internal_pd)   

# shap_values = explainer(X_train_pd.T)   
shap.plots.beeswarm(shap_values, log_scale=False, order=shap.Explanation.abs.mean(0), axis_color='#000000', color=plt.get_cmap("cool"),plot_size=0.2)
plt.subplots_adjust(left=0.125, right=0.995, top=0.975, bottom=0.205)
shap.summary_plot(shap_values,  plot_type="bar")

shap_values1 = explainer(x_external1_pd)   
shap_values2 = explainer(x_external2_pd)   
shap_values3 = explainer(x_external3_pd)   
shap_values4 = explainer(x_external4_pd)   

shap.plots.beeswarm(shap_values1, log_scale=False, order=shap.Explanation.abs.mean(0), axis_color='#000000', color=plt.get_cmap("cool"),plot_size=0.2)
plt.subplots_adjust(left=0.125, right=0.995, top=0.975, bottom=0.205)
shap.summary_plot(shap_values1,  plot_type="bar")

shap.plots.beeswarm(shap_values2, log_scale=False, order=shap.Explanation.abs.mean(0), axis_color='#000000', color=plt.get_cmap("cool"),plot_size=0.2)
plt.subplots_adjust(left=0.125, right=0.995, top=0.975, bottom=0.205)
shap.summary_plot(shap_values2,  plot_type="bar")


#%% Supplementary Fig. 8A-D

model_output_path = r'../model'
MCH = pd.read_csv(os.path.join(proteomic_data_path, r'proteome.txt'), sep='\t')
MCH['group'] = MCH['Category'].str.split('-').str[1]

combination_list = [[('AMBP', )], \
                    [('FN1', )], \
                    [('LUM', )], \
                    [('TIMP1',)], \
                    [('B2M',)], \
                    [('AMBP', 'FN1', 'LUM', 'TIMP1', 'B2M')]]

color_list = ['Blue', '#F8766D', '#7CAE00', '#C77CFF',  '#00BFC4', '#E1AE14']
predict_dict = {}
combine_set_all = []
output_auc = []
count=0
fig, ax = plt.subplots(figsize=(5.5,5.2))

for each_com in combination_list:

    # Train combine
    Train_combine = MCH[MCH['Dataset'].isin(['Train', 'Internal'])]
    
    x_Train_combine_pd = Train_combine[['SampleID'] + list(each_com[0])]
    x_Train_combine_pd = x_Train_combine_pd.set_index('SampleID')
    x_Train_combine_pd = x_Train_combine_pd.div(x_Train_combine_pd.mean(axis=0), axis=1)
    y_Train_combine_pd = Train_combine[['SampleID', 'group']]
    y_Train_combine_pd = y_Train_combine_pd.set_index('SampleID')    
    y_Train_combine = list(chain.from_iterable(enc.fit_transform(y_Train_combine_pd[['group']].values)))
    
    space = {'penalty':['l1', 'l2', 'elasticnet'],  'C': np.logspace(-3, 3, 6),  "solver":['lbfgs',  'saga']}
    model_to_tune = LogisticRegression()
    grid_search=GridSearchCV(model_to_tune, space, scoring="roc_auc", cv=5, n_jobs=-1,  verbose = 1)

    grid_search.fit(x_Train_combine_pd.values, y_Train_combine)
    train_best_model_new = grid_search.best_estimator_
    coef_df_new = pd.DataFrame({'Feature': x_Train_combine_pd.columns, 'Coefficient': train_best_model_new.coef_[0]})
    
    if len(each_com[0]) == 5:
        joblib.dump(train_best_model_new, os.path.join(model_output_path, r'LR_final.pkl'))
    # sklearn2pmml(train_best_model_new, os.path.join(data_path, r'LR_five.pmml'), with_repr=True)

    
    y_Train_combine_pre = train_best_model_new.predict_proba(x_Train_combine_pd.values)
    fpr_Train_combine, tpr_Train_combine, thersholds_Train_combine = roc_curve(y_Train_combine, y_Train_combine_pre[:,1], drop_intermediate=False)
    auc_Train_combine = auc(fpr_Train_combine, tpr_Train_combine)
    
    precision_Train_combine_pr, recall_Train_combine_pr, thresholds_Train_combine_pr = precision_recall_curve(y_Train_combine, y_Train_combine_pre[:,1])
    pr_Train_combine = auc(recall_Train_combine_pr, precision_Train_combine_pr)
    
    N1_train = sum(y_Train_combine)
    N2_train = len(y_Train_combine) - sum(y_Train_combine)
    hanley_ci_train_ = compute_SD(auc_Train_combine, N1_train, N2_train)
    fig, ax = plt.subplots(figsize=(5.5,5.2))
    all_train = auc_plot(thersholds_Train_combine, fpr_Train_combine, tpr_Train_combine, hanley_ci_train_, y_Train_combine_pre, \
                         y_Train_combine, auc_Train_combine, color='dodgerblue', ls='-', label='All', legend=False, auc_plot=True, ax=ax) 
    
    # all External 
    External_all = MCH[MCH['Dataset'].isin(['External1', 'External2', 'External3', 'External4'])]
    x_External_all_pd = External_all[['SampleID'] + list(each_com[0])]
    x_External_all_pd = x_External_all_pd.set_index('SampleID')
    x_External_all_pd = x_External_all_pd.div(x_External_all_pd.mean(axis=0), axis=1)
    y_External_all_pd = External_all[['SampleID', 'group']]
    y_External_all_pd = y_External_all_pd.set_index('SampleID')        
    y_External_all = list(chain.from_iterable(enc.fit_transform(y_External_all_pd[['group']].values)))
    
    
    y_all_pre = train_best_model_new.predict_proba(x_External_all_pd.values)
    fpr_test_all, tpr_test_all, thersholds_all  =roc_curve(y_External_all, y_all_pre[:,1], drop_intermediate=False)
    auc_test_all = auc(fpr_test_all, tpr_test_all)
    
    precision_all_pr, recall_all_pr, thresholds_test_all_pr = precision_recall_curve(y_External_all, y_all_pre[:,1])
    pr_test_all = auc(recall_all_pr, precision_all_pr)
    
    predict_dict[str(each_com)] =  list(y_all_pre[:,1])
   
    N1_all = sum(y_External_all)
    N2_all = len(y_External_all) - sum(y_External_all)
    hanley_ci_all = compute_SD(auc_test_all, N1_all, N2_all)
    pr_hanley_ci_all = compute_SD(pr_test_all, N1_all, N2_all)

    fig, ax = plt.subplots(figsize=(5.5,5.2))
    all_set = auc_plot(thersholds_all, fpr_test_all, tpr_test_all, hanley_ci_all, y_all_pre, \
                         y_External_all, auc_test_all, color=color_list[count], ls='-', label='All', legend=False, auc_plot=True, ax=ax) 
    count+=1
    # fig, ax = plt.subplots(figsize=(5.5,5.2))    
    # all_set_pr = auc_plot(thersholds_all, fpr_test_all, tpr_test_all, pr_hanley_ci_all, y_all_pre, \
    #                      y_External_all, pr_test_all, color='blue', ls='-', label='All', legend=False, auc_plot=False, ax=ax)
    

    output_auc.append([str(each_com[0]), auc_test_all, hanley_ci_all, pr_test_all, pr_hanley_ci_all])


    all_set['combination'] = str(each_com[0])
    
    if type(combine_set_all) == list:
        combine_set_all = all_set
    else:
        combine_set_all = pd.concat([combine_set_all, all_set])    
    

output_auc_pd = pd.DataFrame(output_auc)
output_auc_pd.columns = ['combinations', 'AUROC', 'AUROC_CI', 'AUPR', 'AUPR_CI']


predict = pd.concat([pd.DataFrame(y_Train_combine_pre[:,1]), pd.DataFrame(y_all_pre[:,1])])
predict.index = list(y_Train_combine_pd.index) + list(x_External_all_pd.index)
predict.columns = ['Predict']
predict.to_csv(os.path.join(proteomic_data_path, r'Prediction.txt'), sep='\t')


MCH = pd.merge(MCH, predict, left_on='SampleID', right_index=True)
MCH['subgroup'] = MCH['Category'].str.split('-').str[0]




delong_test = []
for each_com in combination_list[:-1]:
    
    z, p = DelongTest(predict_dict[str(combination_list[-1])], predict_dict[str(each_com)], y_External_all)._show_result()
                
    delong_test.append([str(each_com),z, p])
delong_test_pd = pd.DataFrame(delong_test)
delong_test_pd.columns = ['com', 'z', 'delong']

    



#%% Supplementary Fig. 8F-I
# subgroup
fixed_threshold = 0.2985699122036087

three_category = {'No-risk':['Ctr-FTB', 'Ctr-PTB'],  'Risk':['Risk-FTB', 'Risk-PTB'], 'Symptomatic':['Symptomatic-FTB', 'Symptomatic-PTB']}
three_category_color = {'No-risk':'dodgerblue', 'Risk':'magenta', 'Symptomatic':'red'}

stats_number = []
for category_name, each_category in three_category.items():
    
    category_select = each_category
    
    subgroup_ = MCH[MCH['Category'].isin(category_select)]
    subgroup_train_ = subgroup_[subgroup_['Dataset'].isin(['Train','Internal'])]
    subgroup_test = subgroup_[subgroup_['Dataset'].isin(['External1','External2','External3','External4'])]
    
    
    train_list1 = list(subgroup_train_[subgroup_train_['Category'] == each_category[0]]['SampleID'].values)
    train_list2 = list(subgroup_train_[subgroup_train_['Category'] == each_category[1]]['SampleID'].values)
    
    train_number1 = len(train_list1)
    train_number2 = len(train_list2)
    
    test_list1 = list(subgroup_test[subgroup_test['Category'] == each_category[0]]['SampleID'].values)
    test_list2 = list(subgroup_test[subgroup_test['Category'] == each_category[1]]['SampleID'].values)
    
    test_number1 = len(test_list1)
    test_number2 = len(test_list2)
    
    
    stats_number.append(['Train', each_category[0], train_number1])
    stats_number.append(['Train', each_category[1], train_number2])
    stats_number.append(['Test', each_category[0], test_number1])
    stats_number.append(['Test', each_category[1], test_number2])

stats_number_pd = pd.DataFrame(stats_number)
stats_number_pd.columns=['Dataset', 'Category', 'number']
stats_number_pd = stats_number_pd.set_index(['Dataset', 'Category'])
stats_number_matrix = stats_number_pd['number'].unstack()



matric = []
auc_pr = []
for category_name, each_category in three_category.items():
    
    # category_select = each_category
    # category_select = ['Ctr-FTB', each_category[1]]
    category_select = ['Ctr-FTB']+ each_category
    
    
    subgroup_ = MCH[MCH['Category'].isin(category_select)]
    subgroup_['group'] = subgroup_['Category'].str.split('-').str[1]
        
    subgroup_train_ = subgroup_[subgroup_['Dataset'].isin(['Train','Internal'])]
    
    
    enc = preprocessing.OrdinalEncoder()
    y_subgroup_train_ = subgroup_train_[['group']].values
    y_subgroup_train_ = enc.fit_transform(y_subgroup_train_)
    y_subgroup_train_ = list(chain.from_iterable(y_subgroup_train_))
    
    y_subgroup_train_predict_ = list(subgroup_train_['Predict'].values)
    
    fpr_subgroup_train_, tpr_subgroup_train_, thersholds_subgroup_train_ =roc_curve(y_subgroup_train_, y_subgroup_train_predict_, drop_intermediate=False)
    precision_subgroup_train_pr, recall_subgroup_train_pr, thresholds_subgroup_train_pr = precision_recall_curve(y_subgroup_train_, y_subgroup_train_predict_)
    auc_subgroup_train_ = auc(fpr_subgroup_train_, tpr_subgroup_train_)
    pr_subgroup_train_ = auc(recall_subgroup_train_pr, precision_subgroup_train_pr)
    
    
    N1_subgroup_train_ = sum(y_subgroup_train_)
    N2_subgroup_train_ = len(y_subgroup_train_) - sum(y_subgroup_train_)
    
    hanley_ci_train_auc_ = compute_SD(auc_subgroup_train_, N1_subgroup_train_, N2_subgroup_train_)
    auroc_low_train_ = auc_subgroup_train_-hanley_ci_train_auc_
    auroc_high_train_ = auc_subgroup_train_+hanley_ci_train_auc_
    
    hanley_ci_train_pr_ = compute_SD(pr_subgroup_train_, N1_subgroup_train_, N2_subgroup_train_)
    auprc_low_train_ = pr_subgroup_train_-hanley_ci_train_pr_
    auprc_high_train_ = pr_subgroup_train_+hanley_ci_train_pr_
        
    
    fig, ax = plt.subplots(figsize=(5.5,5.2))
    train_ = auc_plot(thersholds_subgroup_train_, fpr_subgroup_train_, tpr_subgroup_train_, hanley_ci_train_auc_, np.array([[0, i] for i in y_subgroup_train_predict_]),  \
                              y_subgroup_train_, auc_subgroup_train_, color='#00CCFF', ls='-', label='Train', legend=False, auc_plot=True, ax=ax)    
    train_['Category'] = category_name
    
    
    subgroup_test = subgroup_[subgroup_['Dataset'].isin(['External1','External2','External3','External4'])]
    
    enc = preprocessing.OrdinalEncoder()
    y_subgroup_test_ = subgroup_test[['group']].values
    y_subgroup_test_ = enc.fit_transform(y_subgroup_test_)
    y_subgroup_test_ = list(chain.from_iterable(y_subgroup_test_))
    
    y_subgroup_test_predict_ = list(subgroup_test['Predict'].values)

    fpr_subgroup_test_, tpr_subgroup_test_, thersholds_subgroup_test_ =roc_curve(y_subgroup_test_, y_subgroup_test_predict_, drop_intermediate=False)
    precision_subgroup_test_pr, recall_subgroup_test_pr, thresholds_subgroup_test_pr = precision_recall_curve(y_subgroup_test_, y_subgroup_test_predict_)
    auc_subgroup_test_ = auc(fpr_subgroup_test_, tpr_subgroup_test_)
    pr_subgroup_test_ = auc(recall_subgroup_test_pr, precision_subgroup_test_pr)

    
    N1_external_ = sum(y_subgroup_test_)
    N2_external_ = len(y_subgroup_test_) - sum(y_subgroup_test_)
    
    hanley_ci_external_auc_ = compute_SD(auc_subgroup_test_, N1_external_, N2_external_)
    auroc_low_external_ = auc_subgroup_test_-hanley_ci_external_auc_
    auroc_high_external_ = auc_subgroup_test_+hanley_ci_external_auc_
    
    hanley_ci_external_pr_ = compute_SD(pr_subgroup_test_, N1_external_, N2_external_)
    auprc_low_external_ = pr_subgroup_test_-hanley_ci_external_pr_
    auprc_high_external_ = pr_subgroup_test_+hanley_ci_external_pr_
    
    external_ = auc_plot(thersholds_subgroup_test_, fpr_subgroup_test_, tpr_subgroup_test_, hanley_ci_external_pr_, np.array([[0, i] for i in y_subgroup_test_predict_]),  \
                              y_subgroup_test_, auc_subgroup_test_, color='#FF00FF', ls='-', label='Test', legend=False, auc_plot=True, ax=ax)    
    external_['Category'] = category_name
    
    
    metric_ = pd.concat([train_, external_])
    
    auc_pr.append([category_name, auc_subgroup_train_, (f'{auroc_low_train_:.3f}-{auroc_high_train_:.3f}'),
                   pr_subgroup_train_, (f'{auprc_low_train_:.3f}-{auprc_high_train_:.3f}'), 
                   auc_subgroup_test_, (f'{auroc_low_external_:.3f}-{auroc_high_external_:.3f}'),
                   pr_subgroup_test_, (f'{auprc_low_external_:.3f}-{auprc_high_external_:.3f}')])
    
    if type(matric) == list:
        matric = metric_
    else:
        matric = pd.concat([matric, metric_])
    
    # set_max = external_.iloc[external_['F1-score'].idxmax(), :].T
    # plt.plot(1-set_max.loc['Specificity'], set_max.loc['Sensitivity/Recall'], "*", markersize=20, color='red')
    
    plt.savefig(os.path.join(model_output_path, category_name+'_ROC.png'), dpi=300)
    plt.close()    
    
auc_pr_subgroup_df = pd.DataFrame(auc_pr)
auc_pr_subgroup_df.columns = ['Subgroup', 'AUROC_train', 'AUROC_95CI_train', 'AUPRC_train', 'AUPRC_95CI_train', \
                                          'AUROC_external', 'AUROC_95CI_external', 'AUPRC_external', 'AUPRC_95CI_external']




#%%
#subgroup x site

def calculate_calibration(y, p):
    
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)
    vaild = (np.isfinite(y) & np.isfinite(p))
    y = y[vaild]
    p = p[vaild]
    p_clip = np.clip(p, 1e-6, 1 - 1e-6)
    
    brier = brier_score_loss(y, p)
    logit_p = np.log(p_clip/(1-p_clip)).reshape(-1,1)
    
    calibration_model = LogisticRegression(penalty='l2', solver='lbfgs', max_iter=1000)
    calibration_model.fit(logit_p, y)
    
    calibration_intercept = (calibration_model.intercept_[0])
    calibration_slope = (calibration_model.coef_[0, 0])
    
    return {"N":len(y), 'Prevalence':np.mean(y), 'Mean_predicted_risk':np.mean(p), 'Brier_score':brier, 'Calibration_intercept':calibration_intercept, 'Calibration_slope':calibration_slope, 'Status':'Estimable'}



three_category = {'No-risk':['Ctr-FTB', 'Ctr-PTB'],  'Risk':['Risk-FTB', 'Risk-PTB'], 'Symptomatic':['Symptomatic-FTB', 'Symptomatic-PTB']}
three_category_color = {'No-risk':'dodgerblue', 'Risk':'magenta', 'Symptomatic':'red'}
site_definition = {'MCH1': ['Train', 'Internal'], 'MCH2': ['External1'], 'MCH3': ['External2'], 'MCH4': ['External3'], 'MCH5': ['External4']}
MCH['site'] = MCH['SampleID'].str.split('_').str[0]
MCH['group'] = MCH['Category'].str.split('-').str[1]
MCH['Outcome'] = (MCH['Category'].str.endswith('PTB')).astype(int)

fixed_threshold = 0.2985699122036087
# fixed_threshold = cutoff_threshold

site_result = []
for category_name, each_category in three_category.items():
    
    category_select = each_category
    
    subgroup_ = MCH[MCH['Category'].isin(category_select)]
    
    for site_name, dataset_list in site_definition.items():
        
        site_subgroup_ = subgroup_[subgroup_['Dataset'].isin(dataset_list)].copy()
        
        n = len(site_subgroup_)
        n_ptb = int(site_subgroup_['Outcome'].sum())
        n_ftb = n - n_ptb
        
        prevalence = (n_ptb / n if n>0 else np.nan)
        row = {'Subgroup': category_name, 'Site':site_name, 'N':n, 'sPTB':n_ptb, 'FTB':n_ftb, 'Prevalence':prevalence, 'Fix_threshold':fixed_threshold}
        
        if n == 0 or site_subgroup_['Outcome'].nunique() < 2:
            row.update({'AUROC': np.nan, 'AUROC_95CI': np.nan, 'AUPRC': np.nan, 'AUPRC_95CI': np.nan, 'Brier_score': np.nan, 'Calibration_intercept':np.nan, 'Calibration_slope':np.nan, 
                       'TP': np.nan, 'FP': np.nan, 'FN': np.nan, 'TN': np.nan, 'Accuracy': np.nan, 'Sensitivity/Recall': np.nan, 'Specificity': np.nan, 'PPV/Precision': np.nan,
                         'NPV': np.nan, 'F1-score': np.nan, 'False positive rate': np.nan, 'True positive rate': np.nan})
            site_result.append(row)
            continue
        
        y = site_subgroup_['Outcome'].to_numpy()
        p = site_subgroup_['Predict'].to_numpy()
        
        #AUROC
        auroc = roc_auc_score(y, p)
        auroc_hanley_ci_ = compute_SD(auroc, n_ptb, n_ftb)   
        auroc_low_ = auroc - auroc_hanley_ci_
        auroc_high_ = auroc + auroc_hanley_ci_
        
        #AUPR
        auprc = average_precision_score(y, p)
        pr_hanley_ci_ = compute_SD(auprc, n_ptb, n_ftb)   
        auprc_low_ = auprc - pr_hanley_ci_
        auprc_high_ = auprc + pr_hanley_ci_
                
        #Calibration
        brier_score = brier_score_loss(y, p)
        calibration_intercept, calibration_slope = (calculate_calibration(y, p)['Calibration_intercept'], calculate_calibration(y, p)['Calibration_slope'])
        
        #Fixed threshold performance
        fixed_metrics = custom_classification_report(y, [1 if i >= fixed_threshold else 0 for i in p])
        
        row.update({'AUROC': auroc, 'AUROC_95CI': (f'{auroc_low_:.3f}-{auroc_high_:.3f}'), 'AUPRC': auprc, 'AUPRC_95CI': (f'{auprc_low_:.3f}-{auprc_high_:.3f}'),
                    'Brier_score': brier_score, 'Calibration_intercept':calibration_intercept, 'Calibration_slope':calibration_slope, **fixed_metrics})
        
        site_result.append(row)
        
    
column_order = ['Subgroup', 'Site', 'N', 'sPTB','FTB', 'Prevalence',
 'AUROC', 'AUROC_95CI', 'AUPRC', 'AUPRC_95CI', 'Brier_score', 'Calibration_intercept', 'Calibration_slope',
     'TP', 'TN', 'FP', 'FN', 'Sensitivity/Recall', 'Specificity', 'PPV/Precision', 'NPV',
    'F1-score', 'False positive rate', 'True positive rate']


site_result_df = pd.DataFrame(site_result)
site_result_df = site_result_df[column_order]


#%%





