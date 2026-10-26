# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 11:24:40 2024

@author: ZZG
"""




# from sklearnex import patch_sklearn, config_context, unpatch_sklearn
# patch_sklearn('LogisticRegression')
# unpatch_sklearn()
import os
import copy
import time
import shap
import random
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.impute import SimpleImputer
from itertools import chain, combinations
from xgboost import XGBClassifier, XGBRegressor
from scipy.stats import pearsonr, spearmanr
# from thundersvm import SVC
from sklearn import preprocessing
from sklearn.metrics import auc,roc_curve
from sklearn.model_selection import GridSearchCV
# from tune_sklearn import TuneGridSearchCV
from sklearn.linear_model import LinearRegression, ElasticNet, Lasso, Ridge, LogisticRegression, PassiveAggressiveClassifier, SGDClassifier, SGDRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, AdaBoostClassifier, AdaBoostRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier as DNN
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import roc_curve,precision_recall_curve, auc, average_precision_score, confusion_matrix  ###计算roc和auc
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
# from skbio.stats.composition import multiplicative_replacement
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
# from tune_sklearn import TuneGridSearchCV
# from sklearn.experimental import enable_halving_search_cv
# from sklearn.model_selection import HalvingGridSearchCV
from iteration_utilities import random_combination
import stats as st

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



def cal(x, y, flag, protein, formula_str):
    
    if flag == 'up':
        x = x*y
        formula_str = formula_str + '*' + protein
    else:
        x = x/y
        formula_str = formula_str + '/' + protein
    
    return x, formula_str






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



def compute_SD(AUC, PTB, FTB):

    Q1 = AUC/(2-AUC)
    Q2 = 2*AUC*AUC/(1+AUC)
	
    return np.sqrt((AUC*(1-AUC)+(PTB-1)*(Q1-AUC*AUC) + (FTB-1)*(Q2-AUC*AUC))/(PTB*FTB))

def ci_Hanley_McNeil(y_subgroup_, auc_subgroup_, pr_subgroup_):
    
    N1_subgroup_ = sum(y_subgroup_)
    N2_subgroup_ = len(y_subgroup_) - sum(y_subgroup_)
    
    hanley_ci_auc_ = compute_SD(auc_subgroup_, N1_subgroup_, N2_subgroup_)
    auroc_low_= auc_subgroup_-hanley_ci_auc_
    auroc_high_ = auc_subgroup_+hanley_ci_auc_
    
    hanley_ci_pr_ = compute_SD(pr_subgroup_, N1_subgroup_, N2_subgroup_)
    auprc_low_ = pr_subgroup_-hanley_ci_pr_
    auprc_high_ = pr_subgroup_+hanley_ci_pr_
    
    return (auroc_low_, auroc_high_), hanley_ci_auc_, (auprc_low_, auprc_high_), hanley_ci_pr_

proteomic_data_path = r'data'

MCH = pd.read_csv(os.path.join(proteomic_data_path, r'proteome.txt'), sep='\t')
MCH['group'] = MCH['Category'].str.split('-').str[1]
all_combine_pd = MCH[['SampleID'] + ['AMBP', 'FN1', 'LUM', 'TIMP1', 'B2M', 'Cervix length (mm)']].dropna()

enc = preprocessing.OrdinalEncoder()
all_combine_pd = all_combine_pd.set_index('SampleID')
all_sample_list = list(all_combine_pd.index)
y_combine_pd = MCH[MCH['SampleID'].isin(all_sample_list)][['SampleID', 'group']]

# x_Train_combine_pd = x_Train_combine_pd.div(x_Train_combine_pd.mean(axis=0), axis=1)
y_combine_pd = y_combine_pd.set_index('SampleID')    
y_combine_pd = list(chain.from_iterable(enc.fit_transform(y_combine_pd[['group']].values)))

N1_all = sum(y_combine_pd)
N2_all = len(y_combine_pd) - sum(y_combine_pd)


all_combine_pd['1/(CL (mm) +1)'] = 1/(all_combine_pd['Cervix length (mm)']+1)

fpr_cl, tpr_cl, thersholds_cl  =roc_curve(y_combine_pd, all_combine_pd['1/(CL (mm) +1)'], drop_intermediate=False)
auc_cl = auc(fpr_cl, tpr_cl)
precision_cl_pr, recall_cl_pr, thresholds_cl_pr = precision_recall_curve(y_combine_pd, all_combine_pd['1/(CL (mm) +1)'])
pr_cl = auc(recall_cl_pr, precision_cl_pr)   

(auroc_low_cl_, auroc_high_cl_), hanley_ci_cl_auc_, (auprc_low_cl_, auprc_high_cl_), hanley_ci_cl_pr_ = ci_Hanley_McNeil(y_combine_pd, auc_cl, pr_cl)





all_combine_pd = all_combine_pd[['AMBP', 'FN1', 'LUM', 'TIMP1', 'B2M', '1/(CL (mm) +1)']]

space = {'penalty':['l1', 'l2', 'elasticnet'],  'C': np.logspace(-3, 3, 10),  "solver":['lbfgs',  'saga']}
model_to_tune = LogisticRegression()
grid_search=GridSearchCV(model_to_tune, space, scoring="roc_auc", cv=5, n_jobs=-1,  verbose = 1)

grid_search.fit(all_combine_pd.values, y_combine_pd)
best_model_CL = grid_search.best_estimator_
y_new_cl = best_model_CL.predict_proba(all_combine_pd.values)

fpr_model_cl, tpr_model_cl, thersholds_model_cl =roc_curve(y_combine_pd, y_new_cl[:,1], drop_intermediate=False)
precision_model_cl_pr, recall_model_cl_pr, thresholds_model_cl_pr = precision_recall_curve(y_combine_pd, y_new_cl[:,1])
auc_model_cl = auc(fpr_model_cl,tpr_model_cl)
pr_model_cl = auc(recall_model_cl_pr,precision_model_cl_pr)

(auroc_low_cl_model_, auroc_high_cl_model_), hanley_ci_cl_model_auc_, (auprc_low_cl_model_, auprc_high_model_cl_), hanley_ci_cl_model_pr_ = ci_Hanley_McNeil(y_combine_pd, auc_model_cl, pr_model_cl)




fig, ax = plt.subplots(figsize=(5.5,5.2))    


CL_set = auc_plot(thersholds_cl, fpr_cl, tpr_cl, hanley_ci_cl_auc_, np.array([[0, i] for i in all_combine_pd['1/(CL (mm) +1)'].values]), \
                     y_combine_pd, auc_cl, color='gray', ls='-', label='CL', legend=False, auc_plot=True, ax=ax) 

model_set = auc_plot(thersholds_model_cl, fpr_model_cl, tpr_model_cl, hanley_ci_cl_model_auc_, y_new_cl, \
                     y_combine_pd, auc_model_cl, color='#C00000', ls='-', label='Model+CL', legend=False, auc_plot=True, ax=ax) 
    

plt.savefig(os.path.join(proteomic_data_path, 'CL_ROC_All.png'), dpi=300)
plt.close()     

  
z, p = DelongTest(y_new_cl[:,1], all_combine_pd['1/(CL (mm) +1)'], y_combine_pd)._show_result()
                  
                  