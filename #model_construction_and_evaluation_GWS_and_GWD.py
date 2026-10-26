# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 16:45:39 2024

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


output_metric = []
for cutoff_GW_sampling in [34, 30, 28]:
    
    
    MCH = pd.read_csv(os.path.join(proteomic_data_path, r'proteome.txt'), sep='\t')

    MCH['group'] = MCH['Category'].str.split('-').str[1]
    
    prediction = pd.read_csv(os.path.join(proteomic_data_path, r'Prediction.txt'), sep='\t', index_col=0)
    
    MCH = MCH[MCH['Gestational age at sampling, GAS (wks)'] < cutoff_GW_sampling]
    
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
    
    sum(Internal_y+Train_y)
    ptb_number = sum(MCH2_y+MCH3_y+MCH4_y+MCH5_y)
    ftp_number = len(MCH2_y+MCH3_y+MCH4_y+MCH5_y)-sum(MCH2_y+MCH3_y+MCH4_y+MCH5_y)
    
    enc = preprocessing.OrdinalEncoder()
    
    color = ['dodgerblue', 'magenta', '#FFC000', '#74B69F', '#EC748B',]
    fig, ax = plt.subplots(figsize=(5.5,5.2))
    
    for each_com in combination_list:
    
        Train_ = MCH[MCH['Dataset'].isin(['Train','Internal'])]
        y_train_pd = Train_[['SampleID', 'group']]
        y_train_pd = y_train_pd.set_index('SampleID')
        prediction_train_ = prediction.loc[y_train_pd.index]
    
        fpr_train, tpr_train, thersholds_train =roc_curve(enc.fit_transform(y_train_pd), prediction_train_, drop_intermediate=False)
        precision_train_pr, recall_train_pr, thresholds_train_pr = precision_recall_curve(enc.fit_transform(y_train_pd), prediction_train_)
        auc_train = auc(fpr_train,tpr_train)
        pr_train = auc(recall_train_pr,precision_train_pr)
        
        (auroc_low_train_, auroc_high_train_), hanley_ci_train_auc_, (auprc_low_train_, auprc_high_train_), hanley_ci_train_pr_ = ci_Hanley_McNeil(enc.fit_transform(y_train_pd), auc_train, pr_train)
        train_thersholds_output = auc_plot(thersholds_train, fpr_train, tpr_train, hanley_ci_train_auc_[0], np.array([[0, i] for i in prediction_train_['Predict'].values]),  \
                                  list(chain.from_iterable(enc.fit_transform(y_train_pd))), auc_train, color=color[0], ls='-', label='Train', legend=False, auc_plot=True, ax=ax)    
        
        # external1
        y_external1_pd = MCH2[['SampleID', 'group']]
        y_external1_pd = y_external1_pd.set_index('SampleID')
        prediction_external1_ = prediction.loc[y_external1_pd.index]
    
        fpr_external1, tpr_external1, thersholds_external1  =roc_curve(enc.fit_transform(y_external1_pd), prediction_external1_, drop_intermediate=False)
        precision_external1, recall_external1, thresholds_external1_pr = precision_recall_curve(enc.fit_transform(y_external1_pd), prediction_external1_)
        auc_external1 = auc(fpr_external1, tpr_external1)
        pr_external1 = auc(recall_external1, precision_external1)
    
        (auroc_low_external1_, auroc_high_external1_), hanley_ci_external1_auc_, (auprc_low_external1_, auprc_high_external1_), hanley_ci_external1_pr_ = ci_Hanley_McNeil(enc.fit_transform(y_external1_pd), auc_external1, pr_external1)
        external1_thersholds_output = auc_plot(thersholds_external1, fpr_external1, tpr_external1, hanley_ci_external1_auc_[0], np.array([[0, i] for i in prediction_external1_['Predict'].values]),  \
                                  list(chain.from_iterable(enc.fit_transform(y_external1_pd))), auc_external1, color=color[1], ls='-', label='external1', legend=False, auc_plot=True, ax=ax)    
        
        # external2
        y_external2_pd = MCH3[['SampleID', 'group']]
        y_external2_pd = y_external2_pd.set_index('SampleID')
        prediction_external2_ = prediction.loc[y_external2_pd.index]
    
        fpr_external2, tpr_external2, thersholds_external2  =roc_curve(enc.fit_transform(y_external2_pd), prediction_external2_, drop_intermediate=False)
        precision_external2, recall_external2, thresholds_external2_pr = precision_recall_curve(enc.fit_transform(y_external2_pd), prediction_external2_)
        auc_external2 = auc(fpr_external2, tpr_external2)
        pr_external2 = auc(recall_external2, precision_external2)
        
        (auroc_low_external2_, auroc_high_external2_), hanley_ci_external2_auc_, (auprc_low_external2_, auprc_high_external2_), hanley_ci_external2_pr_ = ci_Hanley_McNeil(enc.fit_transform(y_external2_pd), auc_external2, pr_external2)
        external2_thersholds_output = auc_plot(thersholds_external2, fpr_external2, tpr_external2, hanley_ci_external2_auc_[0], np.array([[0, i] for i in prediction_external2_['Predict'].values]),  \
                                  list(chain.from_iterable(enc.fit_transform(y_external2_pd))), auc_external2, color=color[2], ls='-', label='external2', legend=False, auc_plot=True, ax=ax)    
        
    
        # external3
        y_external3_pd = MCH4[['SampleID', 'group']]
        y_external3_pd = y_external3_pd.set_index('SampleID')
        prediction_external3_ = prediction.loc[y_external3_pd.index]
    
        fpr_external3, tpr_external3, thersholds_external3  =roc_curve(enc.fit_transform(y_external3_pd), prediction_external3_, drop_intermediate=False)
        precision_external3, recall_external3, thresholds_external3_pr = precision_recall_curve(enc.fit_transform(y_external3_pd), prediction_external3_)
        auc_external3 = auc(fpr_external3, tpr_external3)
        pr_external3 = auc(recall_external3, precision_external3)
    
        (auroc_low_external3_, auroc_high_external3_), hanley_ci_external3_auc_, (auprc_low_external3_, auprc_high_external3_), hanley_ci_external3_pr_ = ci_Hanley_McNeil(enc.fit_transform(y_external3_pd), auc_external3, pr_external3)
        external3_thersholds_output = auc_plot(thersholds_external3, fpr_external3, tpr_external3, hanley_ci_external3_auc_[0], np.array([[0, i] for i in prediction_external3_['Predict'].values]),  \
                                  list(chain.from_iterable(enc.fit_transform(y_external3_pd))), auc_external3, color=color[3], ls='-', label='external3', legend=False, auc_plot=True, ax=ax)    
        
        # external4
        y_external4_pd = MCH5[['SampleID', 'group']]
        y_external4_pd = y_external4_pd.set_index('SampleID')
        prediction_external4_ = prediction.loc[y_external4_pd.index]
        
        fpr_external4, tpr_external4, thersholds_external4  =roc_curve(enc.fit_transform(y_external4_pd), prediction_external4_, drop_intermediate=False)
        precision_external4, recall_external4, thresholds_external4_pr = precision_recall_curve(enc.fit_transform(y_external4_pd), prediction_external4_)
        auc_external4 = auc(fpr_external4, tpr_external4)
        pr_external4 = auc(recall_external4, precision_external4)
    
        (auroc_low_external4_, auroc_high_external4_), hanley_ci_external4_auc_, (auprc_low_external4_, auprc_high_external4_), hanley_ci_external4_pr_ = ci_Hanley_McNeil(enc.fit_transform(y_external4_pd), auc_external4, pr_external4)
        external4_thersholds_output = auc_plot(thersholds_external4, fpr_external4, tpr_external4, hanley_ci_external4_auc_[0], np.array([[0, i] for i in prediction_external4_['Predict'].values]),  \
                                  list(chain.from_iterable(enc.fit_transform(y_external4_pd))), auc_external4, color=color[4], ls='-', label='external4', legend=False, auc_plot=True, ax=ax)    
            
    
        plt.savefig(os.path.join(proteomic_data_path, 'Sampling_'+str(cutoff_GW_sampling)+'_ROC_MCH.png'), dpi=300)
        plt.close()  
    
        fig, ax = plt.subplots(figsize=(5.5,5.2))
        # all_external
        y_external_all = pd.concat([y_external1_pd, y_external2_pd, y_external3_pd, y_external4_pd])
        prediction_external_all = prediction.loc[y_external_all.index]
        
        fpr_external, tpr_external, thersholds_external  =roc_curve(enc.fit_transform(y_external_all), prediction_external_all, drop_intermediate=False)
        precision_external, recall_external, thresholds_external_pr = precision_recall_curve(enc.fit_transform(y_external_all), prediction_external_all)
        auc_external = auc(fpr_external, tpr_external)
        pr_external = auc(recall_external, precision_external)
        
        (auroc_low_external_, auroc_high_external_), hanley_ci_external_auc_, (auprc_low_external_, auprc_high_external_), hanley_ci_external_pr_ = ci_Hanley_McNeil(enc.fit_transform(y_external_all), auc_external, pr_external)
        external4_thersholds_output = auc_plot(thersholds_external, fpr_external, tpr_external, hanley_ci_external_auc_[0], np.array([[0, i] for i in prediction_external_all['Predict'].values]),  \
                                  list(chain.from_iterable(enc.fit_transform(y_external_all))), auc_external, color='blue', ls='-', label='external_all', legend=False, auc_plot=True, ax=ax)    
        plt.savefig(os.path.join(proteomic_data_path, 'Sampling_'+str(cutoff_GW_sampling)+'_ROC_All.png'), dpi=300)
        plt.close()           
        
        
    fixed_threshold = 0.2985699122036087    
    y = list(chain.from_iterable(enc.fit_transform(y_external_all)))
    fixed_metrics = custom_classification_report(y, [1 if i >= fixed_threshold else 0 for i in prediction_external_all['Predict'].values])
    
    output_metric.append([auc_train, (auroc_low_train_, auroc_high_train_), auc_external1,  (auroc_low_external1_, auroc_high_external1_),
                          auc_external2,  (auroc_low_external2_, auroc_high_external2_), auc_external3,  (auroc_low_external3_, auroc_high_external3_),
                          auc_external4,  (auroc_low_external4_, auroc_high_external4_), 
                          ptb_number, ftp_number, auc_external, (auroc_low_external_, auroc_high_external_),
        fixed_metrics['Specificity']+fixed_metrics['Sensitivity/Recall']-1, fixed_metrics['Sensitivity/Recall'], fixed_metrics['Specificity'], fixed_metrics['PPV/Precision'], fixed_metrics['NPV']])

output_metric_pd = pd.DataFrame(output_metric)
output_metric_pd.columns = ['AUC_train', 'AUC_train_CI', 'AUC_external1', 'AUC_external1_CI', 'AUC_external2', 'AUC_external2_CI',  'AUC_external3', 'AUC_external3_CI',\
                            'AUC_external4', 'AUC_external4_CI', 'PTB_number', 'FTB_number', 'AUC_external', 'AUC_external_CI', 'Youden', 'Sensitivity', 'Specificity', 'PPV', 'NPV',]



#%%



output_metric_delivery = []
for cutoff_GW_delivery in [34, 33, 30]:
    
    MCH = pd.read_csv(os.path.join(proteomic_data_path, r'proteome.txt'), sep='\t')

    MCH['group'] = MCH['Category'].str.split('-').str[1]
    
    prediction = pd.read_csv(os.path.join(proteomic_data_path, r'Prediction.txt'), sep='\t', index_col=0)
    
    clinic_info_all = []
    for each in [['Train', 'Internal'], ['External1'], ['External2'], ['External3'], ['External4']]:
        MCH_ = MCH[MCH['Dataset'].isin(each)]
        clinic_info_PTB = MCH_[MCH_['Gestational age at birth, GAB (wks)'] < cutoff_GW_delivery]
        clinic_info_ctr = MCH_[((MCH_['Gestational age at birth, GAB (wks)']>=37) & (MCH_['Gestational age at sampling, GAS (wks)'] <= clinic_info_PTB['Gestational age at sampling, GAS (wks)'].max()))]
        clinic_info = pd.concat([clinic_info_PTB[['Category']], clinic_info_ctr[['Category']]])
        
        if type(clinic_info_all) == list:
            clinic_info_all = clinic_info
        else:
            clinic_info_all = pd.concat([clinic_info_all, clinic_info])
        
    MCH = MCH.loc[clinic_info_all.index]
    
    
    
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
    
    sum(Internal_y+Train_y)
    ptb_number = sum(MCH2_y+MCH3_y+MCH4_y+MCH5_y)
    ftp_number = len(MCH2_y+MCH3_y+MCH4_y+MCH5_y)-sum(MCH2_y+MCH3_y+MCH4_y+MCH5_y)
    
    
    enc = preprocessing.OrdinalEncoder()
    
    color = ['dodgerblue', 'magenta', '#FFC000', '#74B69F', '#EC748B',]
    fig, ax = plt.subplots(figsize=(5.5,5.2))
    
    for each_com in combination_list:
    
        Train_ = MCH[MCH['Dataset'].isin(['Train','Internal'])]
        y_train_pd = Train_[['SampleID', 'group']]
        y_train_pd = y_train_pd.set_index('SampleID')
        prediction_train_ = prediction.loc[y_train_pd.index]
    
        fpr_train, tpr_train, thersholds_train =roc_curve(enc.fit_transform(y_train_pd), prediction_train_, drop_intermediate=False)
        precision_train_pr, recall_train_pr, thresholds_train_pr = precision_recall_curve(enc.fit_transform(y_train_pd), prediction_train_)
        auc_train = auc(fpr_train,tpr_train)
        pr_train = auc(recall_train_pr,precision_train_pr)
        
        (auroc_low_train_, auroc_high_train_), hanley_ci_train_auc_, (auprc_low_train_, auprc_high_train_), hanley_ci_train_pr_ = ci_Hanley_McNeil(enc.fit_transform(y_train_pd), auc_train, pr_train)
        train_thersholds_output = auc_plot(thersholds_train, fpr_train, tpr_train, hanley_ci_train_auc_[0], np.array([[0, i] for i in prediction_train_['Predict'].values]),  \
                                  list(chain.from_iterable(enc.fit_transform(y_train_pd))), auc_train, color=color[0], ls='-', label='Train', legend=False, auc_plot=True, ax=ax)    
        
        # external1
        y_external1_pd = MCH2[['SampleID', 'group']]
        y_external1_pd = y_external1_pd.set_index('SampleID')
        prediction_external1_ = prediction.loc[y_external1_pd.index]
    
        fpr_external1, tpr_external1, thersholds_external1  =roc_curve(enc.fit_transform(y_external1_pd), prediction_external1_, drop_intermediate=False)
        precision_external1, recall_external1, thresholds_external1_pr = precision_recall_curve(enc.fit_transform(y_external1_pd), prediction_external1_)
        auc_external1 = auc(fpr_external1, tpr_external1)
        pr_external1 = auc(recall_external1, precision_external1)
    
        (auroc_low_external1_, auroc_high_external1_), hanley_ci_external1_auc_, (auprc_low_external1_, auprc_high_external1_), hanley_ci_external1_pr_ = ci_Hanley_McNeil(enc.fit_transform(y_external1_pd), auc_external1, pr_external1)
        external1_thersholds_output = auc_plot(thersholds_external1, fpr_external1, tpr_external1, hanley_ci_external1_auc_[0], np.array([[0, i] for i in prediction_external1_['Predict'].values]),  \
                                  list(chain.from_iterable(enc.fit_transform(y_external1_pd))), auc_external1, color=color[1], ls='-', label='external1', legend=False, auc_plot=True, ax=ax)    
        
        # external2
        y_external2_pd = MCH3[['SampleID', 'group']]
        y_external2_pd = y_external2_pd.set_index('SampleID')
        prediction_external2_ = prediction.loc[y_external2_pd.index]
    
        fpr_external2, tpr_external2, thersholds_external2  =roc_curve(enc.fit_transform(y_external2_pd), prediction_external2_, drop_intermediate=False)
        precision_external2, recall_external2, thresholds_external2_pr = precision_recall_curve(enc.fit_transform(y_external2_pd), prediction_external2_)
        auc_external2 = auc(fpr_external2, tpr_external2)
        pr_external2 = auc(recall_external2, precision_external2)
        
        (auroc_low_external2_, auroc_high_external2_), hanley_ci_external2_auc_, (auprc_low_external2_, auprc_high_external2_), hanley_ci_external2_pr_ = ci_Hanley_McNeil(enc.fit_transform(y_external2_pd), auc_external2, pr_external2)
        external2_thersholds_output = auc_plot(thersholds_external2, fpr_external2, tpr_external2, hanley_ci_external2_auc_[0], np.array([[0, i] for i in prediction_external2_['Predict'].values]),  \
                                  list(chain.from_iterable(enc.fit_transform(y_external2_pd))), auc_external2, color=color[2], ls='-', label='external2', legend=False, auc_plot=True, ax=ax)    
        
    
        # external3
        y_external3_pd = MCH4[['SampleID', 'group']]
        y_external3_pd = y_external3_pd.set_index('SampleID')
        prediction_external3_ = prediction.loc[y_external3_pd.index]
    
        fpr_external3, tpr_external3, thersholds_external3  =roc_curve(enc.fit_transform(y_external3_pd), prediction_external3_, drop_intermediate=False)
        precision_external3, recall_external3, thresholds_external3_pr = precision_recall_curve(enc.fit_transform(y_external3_pd), prediction_external3_)
        auc_external3 = auc(fpr_external3, tpr_external3)
        pr_external3 = auc(recall_external3, precision_external3)
    
        (auroc_low_external3_, auroc_high_external3_), hanley_ci_external3_auc_, (auprc_low_external3_, auprc_high_external3_), hanley_ci_external3_pr_ = ci_Hanley_McNeil(enc.fit_transform(y_external3_pd), auc_external3, pr_external3)
        external3_thersholds_output = auc_plot(thersholds_external3, fpr_external3, tpr_external3, hanley_ci_external3_auc_[0], np.array([[0, i] for i in prediction_external3_['Predict'].values]),  \
                                  list(chain.from_iterable(enc.fit_transform(y_external3_pd))), auc_external3, color=color[3], ls='-', label='external3', legend=False, auc_plot=True, ax=ax)    
        
        # external4
        y_external4_pd = MCH5[['SampleID', 'group']]
        y_external4_pd = y_external4_pd.set_index('SampleID')
        prediction_external4_ = prediction.loc[y_external4_pd.index]
        
        fpr_external4, tpr_external4, thersholds_external4  =roc_curve(enc.fit_transform(y_external4_pd), prediction_external4_, drop_intermediate=False)
        precision_external4, recall_external4, thresholds_external4_pr = precision_recall_curve(enc.fit_transform(y_external4_pd), prediction_external4_)
        auc_external4 = auc(fpr_external4, tpr_external4)
        pr_external4 = auc(recall_external4, precision_external4)
    
        (auroc_low_external4_, auroc_high_external4_), hanley_ci_external4_auc_, (auprc_low_external4_, auprc_high_external4_), hanley_ci_external4_pr_ = ci_Hanley_McNeil(enc.fit_transform(y_external4_pd), auc_external4, pr_external4)
        external4_thersholds_output = auc_plot(thersholds_external4, fpr_external4, tpr_external4, hanley_ci_external4_auc_[0], np.array([[0, i] for i in prediction_external4_['Predict'].values]),  \
                                  list(chain.from_iterable(enc.fit_transform(y_external4_pd))), auc_external4, color=color[4], ls='-', label='external4', legend=False, auc_plot=True, ax=ax)    
            
    
        plt.savefig(os.path.join(proteomic_data_path, 'Delivery_'+str(cutoff_GW_delivery)+'_ROC_MCH.png'), dpi=300)
        plt.close()  
    
        fig, ax = plt.subplots(figsize=(5.5,5.2))
        # all_external
        y_external_all = pd.concat([y_external1_pd, y_external2_pd, y_external3_pd, y_external4_pd])
        prediction_external_all = prediction.loc[y_external_all.index]
        
        fpr_external, tpr_external, thersholds_external  =roc_curve(enc.fit_transform(y_external_all), prediction_external_all, drop_intermediate=False)
        precision_external, recall_external, thresholds_external_pr = precision_recall_curve(enc.fit_transform(y_external_all), prediction_external_all)
        auc_external = auc(fpr_external, tpr_external)
        pr_external = auc(recall_external, precision_external)
        
        (auroc_low_external_, auroc_high_external_), hanley_ci_external_auc_, (auprc_low_external_, auprc_high_external_), hanley_ci_external_pr_ = ci_Hanley_McNeil(enc.fit_transform(y_external_all), auc_external, pr_external)
        external_all_thersholds_output = auc_plot(thersholds_external, fpr_external, tpr_external, hanley_ci_external_auc_[0], np.array([[0, i] for i in prediction_external_all['Predict'].values]),  \
                                  list(chain.from_iterable(enc.fit_transform(y_external_all))), auc_external, color='blue', ls='-', label='external_all', legend=False, auc_plot=True, ax=ax)    
        plt.savefig(os.path.join(proteomic_data_path, 'Delivery_'+str(cutoff_GW_delivery)+'_ROC_All.png'), dpi=300)
        plt.close()           
        
        
    fixed_threshold = 0.2985699122036087    
    y = list(chain.from_iterable(enc.fit_transform(y_external_all)))
    fixed_metrics = custom_classification_report(y, [1 if i >= fixed_threshold else 0 for i in prediction_external_all['Predict'].values])
    
    output_metric_delivery.append([auc_train, (auroc_low_train_, auroc_high_train_), auc_external1,  (auroc_low_external1_, auroc_high_external1_),
                          auc_external2,  (auroc_low_external2_, auroc_high_external2_), auc_external3,  (auroc_low_external3_, auroc_high_external3_),
                          auc_external4,  (auroc_low_external4_, auroc_high_external4_), 
                          ptb_number, ftp_number, auc_external, (auroc_low_external_, auroc_high_external_),
                          fixed_metrics['Specificity']+fixed_metrics['Sensitivity/Recall']-1, fixed_metrics['Sensitivity/Recall'], fixed_metrics['Specificity'], fixed_metrics['PPV/Precision'], fixed_metrics['NPV']])

output_metric_delivery_pd = pd.DataFrame(output_metric_delivery)
output_metric_delivery_pd.columns = ['AUC_train', 'AUC_train_CI', 'AUC_external1', 'AUC_external1_CI', 'AUC_external2', 'AUC_external2_CI',  'AUC_external3', 'AUC_external3_CI',\
                            'AUC_external4', 'AUC_external4_CI', 'PTB_number', 'FTB_number', 'AUC_external', 'AUC_external_CI', 'Youden', 'Sensitivity', 'Specificity', 'PPV', 'NPV',]


    
#%%


day_interval = 7

def interval_category(clinic_info, PTB_list, Ctr_list, day=7, category = 'interval', new_category='group'):
    
    new_clinic_info = copy.deepcopy(clinic_info)
    
    cutoff_interval = day/7
    clinic_info_PTB = new_clinic_info.loc[PTB_list]
    clinic_info_PTB_filter = clinic_info_PTB[clinic_info_PTB[category] > cutoff_interval]
    clinic_info_PTB_remain = clinic_info_PTB[clinic_info_PTB[category] <= cutoff_interval]
    clinic_info_PTB_filter = copy.deepcopy(clinic_info_PTB_filter)
    clinic_info_PTB_filter[new_category] = "FTB"
    clinic_info_Ctr = new_clinic_info.loc[Ctr_list]

    clinic_info_combine = pd.concat([clinic_info_PTB_filter[[new_category]],  clinic_info_PTB_remain[[new_category]],  clinic_info_Ctr[[new_category]]])
    
    return clinic_info_combine

    

color_dict = {7:'#FFC000', 14:'#00CC00', 28:'#0099FF'}
fig_external, ax_external = plt.subplots(figsize=(5.5,5.2))

output_threshold = []

output_metric_interval = []
for day_interval in [7, 14, 28]:
    
    MCH = pd.read_csv(os.path.join(proteomic_data_path, r'proteome.txt'), sep='\t')

    MCH['group'] = MCH['Category'].str.split('-').str[1]
    
    MCH_clinic = MCH.iloc[:, :9]
    MCH_clinic['interval'] = MCH_clinic['Gestational age at birth, GAB (wks)'] - MCH_clinic['Gestational age at sampling, GAS (wks)']
    MCH_clinic['group'] = MCH_clinic['Category'].str.split('-').str[1]
    MCH_clinic = MCH_clinic.set_index('SampleID')
    PTB_list = list(MCH[MCH['group'] == 'PTB']['SampleID'])
    Ctr_list = list(MCH[MCH['group'] == 'FTB']['SampleID'])
    group = interval_category(MCH_clinic, PTB_list, Ctr_list, day=day_interval, category = 'interval', new_category='group')
    
    
    prediction = pd.read_csv(os.path.join(proteomic_data_path, r'Prediction.txt'), sep='\t', index_col=0)
    
    MCH['group'] =  group.loc[MCH['SampleID']]['group'].values
    
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
    
    sum(Internal_y+Train_y)
    ptb_number = sum(MCH2_y+MCH3_y+MCH4_y+MCH5_y)
    ftp_number = len(MCH2_y+MCH3_y+MCH4_y+MCH5_y)-sum(MCH2_y+MCH3_y+MCH4_y+MCH5_y)
    
    
    enc = preprocessing.OrdinalEncoder()
    
    # color = ['dodgerblue', 'magenta', '#FFC000', '#74B69F', '#EC748B',]
    
    # fig, ax = plt.subplots(figsize=(5.5,5.2))
    for each_com in combination_list:
    
        Train_ = MCH[MCH['Dataset'].isin(['Train','Internal'])]
        y_train_pd = Train_[['SampleID', 'group']]
        y_train_pd = y_train_pd.set_index('SampleID')
        prediction_train_ = prediction.loc[y_train_pd.index]
    
        fpr_train, tpr_train, thersholds_train =roc_curve(enc.fit_transform(y_train_pd), prediction_train_, drop_intermediate=False)
        precision_train_pr, recall_train_pr, thresholds_train_pr = precision_recall_curve(enc.fit_transform(y_train_pd), prediction_train_)
        auc_train = auc(fpr_train,tpr_train)
        pr_train = auc(recall_train_pr,precision_train_pr)
        
        (auroc_low_train_, auroc_high_train_), hanley_ci_train_auc_, (auprc_low_train_, auprc_high_train_), hanley_ci_train_pr_ = ci_Hanley_McNeil(enc.fit_transform(y_train_pd), auc_train, pr_train)
        # train_thersholds_output = auc_plot(thersholds_train, fpr_train, tpr_train, hanley_ci_train_auc_[0], np.array([[0, i] for i in prediction_train_['Predict'].values]),  \
        #                           list(chain.from_iterable(enc.fit_transform(y_train_pd))), auc_train, color=color[0], ls='-', label='Train', legend=False, auc_plot=True, ax=ax)    
        
        # external1
        y_external1_pd = MCH2[['SampleID', 'group']]
        y_external1_pd = y_external1_pd.set_index('SampleID')
        prediction_external1_ = prediction.loc[y_external1_pd.index]
    
        fpr_external1, tpr_external1, thersholds_external1  =roc_curve(enc.fit_transform(y_external1_pd), prediction_external1_, drop_intermediate=False)
        precision_external1, recall_external1, thresholds_external1_pr = precision_recall_curve(enc.fit_transform(y_external1_pd), prediction_external1_)
        auc_external1 = auc(fpr_external1, tpr_external1)
        pr_external1 = auc(recall_external1, precision_external1)
    
        (auroc_low_external1_, auroc_high_external1_), hanley_ci_external1_auc_, (auprc_low_external1_, auprc_high_external1_), hanley_ci_external1_pr_ = ci_Hanley_McNeil(enc.fit_transform(y_external1_pd), auc_external1, pr_external1)
        # external1_thersholds_output = auc_plot(thersholds_external1, fpr_external1, tpr_external1, hanley_ci_external1_auc_[0], np.array([[0, i] for i in prediction_external1_['Predict'].values]),  \
        #                           list(chain.from_iterable(enc.fit_transform(y_external1_pd))), auc_external1, color=color[1], ls='-', label='external1', legend=False, auc_plot=True, ax=ax)    
        
        # external2
        y_external2_pd = MCH3[['SampleID', 'group']]
        y_external2_pd = y_external2_pd.set_index('SampleID')
        prediction_external2_ = prediction.loc[y_external2_pd.index]
    
        fpr_external2, tpr_external2, thersholds_external2  =roc_curve(enc.fit_transform(y_external2_pd), prediction_external2_, drop_intermediate=False)
        precision_external2, recall_external2, thresholds_external2_pr = precision_recall_curve(enc.fit_transform(y_external2_pd), prediction_external2_)
        auc_external2 = auc(fpr_external2, tpr_external2)
        pr_external2 = auc(recall_external2, precision_external2)
        
        (auroc_low_external2_, auroc_high_external2_), hanley_ci_external2_auc_, (auprc_low_external2_, auprc_high_external2_), hanley_ci_external2_pr_ = ci_Hanley_McNeil(enc.fit_transform(y_external2_pd), auc_external2, pr_external2)
        # external2_thersholds_output = auc_plot(thersholds_external2, fpr_external2, tpr_external2, hanley_ci_external2_auc_[0], np.array([[0, i] for i in prediction_external2_['Predict'].values]),  \
        #                           list(chain.from_iterable(enc.fit_transform(y_external2_pd))), auc_external2, color=color[2], ls='-', label='external2', legend=False, auc_plot=True, ax=ax)    
        
    
        # external3
        y_external3_pd = MCH4[['SampleID', 'group']]
        y_external3_pd = y_external3_pd.set_index('SampleID')
        prediction_external3_ = prediction.loc[y_external3_pd.index]
    
        fpr_external3, tpr_external3, thersholds_external3  =roc_curve(enc.fit_transform(y_external3_pd), prediction_external3_, drop_intermediate=False)
        precision_external3, recall_external3, thresholds_external3_pr = precision_recall_curve(enc.fit_transform(y_external3_pd), prediction_external3_)
        auc_external3 = auc(fpr_external3, tpr_external3)
        pr_external3 = auc(recall_external3, precision_external3)
    
        (auroc_low_external3_, auroc_high_external3_), hanley_ci_external3_auc_, (auprc_low_external3_, auprc_high_external3_), hanley_ci_external3_pr_ = ci_Hanley_McNeil(enc.fit_transform(y_external3_pd), auc_external3, pr_external3)
        # external3_thersholds_output = auc_plot(thersholds_external3, fpr_external3, tpr_external3, hanley_ci_external3_auc_[0], np.array([[0, i] for i in prediction_external3_['Predict'].values]),  \
        #                           list(chain.from_iterable(enc.fit_transform(y_external3_pd))), auc_external3, color=color[3], ls='-', label='external3', legend=False, auc_plot=True, ax=ax)    
        
        # external4
        y_external4_pd = MCH5[['SampleID', 'group']]
        y_external4_pd = y_external4_pd.set_index('SampleID')
        prediction_external4_ = prediction.loc[y_external4_pd.index]
        
        fpr_external4, tpr_external4, thersholds_external4  =roc_curve(enc.fit_transform(y_external4_pd), prediction_external4_, drop_intermediate=False)
        precision_external4, recall_external4, thresholds_external4_pr = precision_recall_curve(enc.fit_transform(y_external4_pd), prediction_external4_)
        auc_external4 = auc(fpr_external4, tpr_external4)
        pr_external4 = auc(recall_external4, precision_external4)
    
        (auroc_low_external4_, auroc_high_external4_), hanley_ci_external4_auc_, (auprc_low_external4_, auprc_high_external4_), hanley_ci_external4_pr_ = ci_Hanley_McNeil(enc.fit_transform(y_external4_pd), auc_external4, pr_external4)
        # external4_thersholds_output = auc_plot(thersholds_external4, fpr_external4, tpr_external4, hanley_ci_external4_auc_[0], np.array([[0, i] for i in prediction_external4_['Predict'].values]),  \
        #                           list(chain.from_iterable(enc.fit_transform(y_external4_pd))), auc_external4, color=color[4], ls='-', label='external4', legend=False, auc_plot=True, ax=ax)    
            
    
        # plt.savefig(os.path.join(proteomic_data_path, 'Interval_'+str(cutoff_GW_delivery)+'_ROC_MCH.png'), dpi=300)
        # plt.close()  
    
        
        # all_external
        y_external_all = pd.concat([y_external1_pd, y_external2_pd, y_external3_pd, y_external4_pd])
        prediction_external_all = prediction.loc[y_external_all.index]
        
        fpr_external, tpr_external, thersholds_external  =roc_curve(enc.fit_transform(y_external_all), prediction_external_all, drop_intermediate=False)
        precision_external, recall_external, thresholds_external_pr = precision_recall_curve(enc.fit_transform(y_external_all), prediction_external_all)
        auc_external = auc(fpr_external, tpr_external)
        pr_external = auc(recall_external, precision_external)
        
        (auroc_low_external_, auroc_high_external_), hanley_ci_external_auc_, (auprc_low_external_, auprc_high_external_), hanley_ci_external_pr_ = ci_Hanley_McNeil(enc.fit_transform(y_external_all), auc_external, pr_external)
        external_all_thersholds_output = auc_plot(thersholds_external, fpr_external, tpr_external, hanley_ci_external_auc_[0], np.array([[0, i] for i in prediction_external_all['Predict'].values]),  \
                                  list(chain.from_iterable(enc.fit_transform(y_external_all))), auc_external, color=color_dict[day_interval], ls='-', label='external_all', legend=False, auc_plot=True, ax=ax_external)    
        
        external_all_thersholds_output['Prediction windows'] = day_interval
        if type(output_threshold) == list:
            output_threshold = external_all_thersholds_output
        else:
            output_threshold = pd.concat([output_threshold, external_all_thersholds_output])
        
    fixed_threshold = 0.2985699122036087    
    y = list(chain.from_iterable(enc.fit_transform(y_external_all)))
    fixed_metrics = custom_classification_report(y, [1 if i >= fixed_threshold else 0 for i in prediction_external_all['Predict'].values])
    
    output_metric_interval.append(['Interval_'+str(day_interval), auc_train, (auroc_low_train_, auroc_high_train_), auc_external1,  (auroc_low_external1_, auroc_high_external1_),
                          auc_external2,  (auroc_low_external2_, auroc_high_external2_), auc_external3,  (auroc_low_external3_, auroc_high_external3_),
                          auc_external4,  (auroc_low_external4_, auroc_high_external4_), 
                          ptb_number, ftp_number, auc_external, (auroc_low_external_, auroc_high_external_),
                          fixed_metrics['Specificity']+fixed_metrics['Sensitivity/Recall']-1, fixed_metrics['Sensitivity/Recall'], fixed_metrics['Specificity'], fixed_metrics['PPV/Precision'], fixed_metrics['NPV']])

output_metric_interval_pd = pd.DataFrame(output_metric_interval)
output_metric_interval_pd.columns = ['Interval', 'AUC_train', 'AUC_train_CI', 'AUC_external1', 'AUC_external1_CI', 'AUC_external2', 'AUC_external2_CI',  'AUC_external3', 'AUC_external3_CI',\
                            'AUC_external4', 'AUC_external4_CI', 'PTB_number', 'FTB_number', 'AUC_external', 'AUC_external_CI', 'Youden', 'Sensitivity', 'Specificity', 'PPV', 'NPV',]

plt.savefig(os.path.join(proteomic_data_path, 'Interval_'+str(cutoff_GW_delivery)+'_ROC_All.png'), dpi=300)
plt.close()       

#%%
         