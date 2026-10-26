# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 11:30:34 2023

@author: zhaozhiguang1
"""

#Python 3.11.5
import os
import random
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as st
import matplotlib.pyplot as plt
from collections import Counter
from scipy.stats import mannwhitneyu
import calculate_tools.fc as foldchange
from statsmodels.sandbox.stats.multicomp import multipletests



def pvalue_adjust(pvalue_matrix):
    # multiple-test adjustment
    mask_all = np.isfinite(pvalue_matrix.values)
    pval_corrected_all = np.full(pvalue_matrix.shape, np.nan)
    pval_corrected_all[mask_all] = multipletests(pvalue_matrix.values[mask_all], method='fdr_bh')[1]
    pval_corrected_all_df = pd.DataFrame(pval_corrected_all, index=pvalue_matrix.index, columns = pvalue_matrix.columns)
    return pval_corrected_all_df


def deg(expression_matrix, clin, category, category_list, method='mean'):
    # differential expression analysis
    all_stat_data = []    
    expression_matrix_isna = np.isnan(expression_matrix)
    
    for each_protein in expression_matrix.index:
        
        protein_ = expression_matrix.loc[each_protein]
        proteome_raw_isna_ = expression_matrix_isna.loc[each_protein]
        
        protein_pd_ = pd.concat([protein_, clin], axis=1)
    
    
        ctr = protein_pd_[(protein_pd_[category] == category_list[0])]
        ptb = protein_pd_[(protein_pd_[category] == category_list[1])]
        ctr['category'] = category_list[0]
        ptb['category'] = category_list[1]
        
        if method == 'mean':
            ptb_mean_ = ptb[each_protein].mean()
            ctr_mean_ = ctr[each_protein].mean()
        else:
            
            ptb_mean_ = ptb[each_protein].median()
            ctr_mean_ = ctr[each_protein].median()
        
        fc_, flag = foldchange.fc(ptb_mean_, ctr_mean_)
        
        proteome_raw_isna_ctr_ = 1-proteome_raw_isna_.loc[ctr.index].mean()
        proteome_raw_isna_ptb_ = 1-proteome_raw_isna_.loc[ptb.index].mean()
        
        value_ptb = ptb[each_protein].dropna()
        value_ctr = ctr[each_protein].dropna()
        
        try:
            u_, p_ = mannwhitneyu(list(value_ptb), list(value_ctr))
        except:
            p_ = np.nan
            
        all_stat_data.append([each_protein, len(value_ctr),len(value_ptb),  \
                              proteome_raw_isna_ctr_, proteome_raw_isna_ptb_,  \
                              ctr_mean_, ptb_mean_,  fc_, flag, p_ ])
        
        
    all_stat_data_pd = pd.DataFrame(all_stat_data)
    all_stat_data_pd.columns = ['protein', category_list[0]+'_number', category_list[1]+'_number', category_list[0]+'_ratio', category_list[1]+'_ratio', \
                                category_list[0], category_list[1], 'FC', 'up/down', 'pvalue']
    all_stat_data_pd['qvalue'] = pvalue_adjust(all_stat_data_pd[['pvalue']])
    
    return all_stat_data_pd



def count_elements_in_row(row):
    counts = Counter(row)
    return pd.Series(counts)


plt.rcParams['font.size'] = 18
plt.rcParams['font.family'] = 'arial'


working_dir = r'..'
os.chdir(working_dir)


proteome_train = pd.read_csv(r'path_of_training_set_datamatrix', sep='\t', index_col=0)
# example of data format
'''
                A1BG       A2M     A2ML1  ...      ZW10        ZYX     ZZEF1
SampleID1     0.328661  0.129112  0.806182  ...  0.613226   0.372298  0.441635
SampleID2   0.677768  0.408715  6.885318  ...  0.276755   0.296675  0.135660
SampleID3     0.348017  0.069722  1.337770  ...  1.046826   1.191121  0.633198
SampleID4     0.804198  2.184097  1.155213  ...  1.000000   1.000000  1.000000
SampleID5    22.684784  1.066809  0.185037  ...  1.129493   0.718909  0.592854
'''

clinic_info = pd.read_csv(r'group_file_of_training_set_samples', sep='\t', index_col=0)
# example of data format
'''
          Group      
SampleID1          FTB  
SampleID2        PTB  
SampleID3          FTB 
SampleID4          FTB  
SampleID5          PTB
'''

clinic_info = clinic_info.loc[list(proteome_train.index)]


# stats NA ratio at different group
proteome_train_isna = np.isnan(proteome_train)
ftb_list = clinic_info[clinic_info['Group'] == 'FTB'].index
ptb_list = clinic_info[clinic_info['Group'] == 'PTB'].index
proteome_train_isna_group = pd.concat([1-proteome_train_isna.loc[ftb_list].mean(), 1-proteome_train_isna.loc[ptb_list].mean()], axis=1)
proteome_train_isna_group.columns = ['FTB_ratio', 'PTB_ratio']



#%% jackknifing

#Initialize the output result list
direction = []
pvalue = []
qvalue = []
FC = []


sample_list = list(proteome_train.columns)

# Specify the number of samples for each random sampling
number_random = int(np.round(len(sample_list)*0.9, 0))

#counting
count = 0
for each in np.arange(1000):
    
    sample_list_random = list(random.sample(sample_list, number_random))
    proteome_random = proteome_train[sample_list_random]
    clinic_info_random = clinic_info.loc[sample_list_random]
    
    # Differential protein analysis
    stat = deg(proteome_random, clinic_info_random, 'Group', ['FTB', 'PTB'])
    
    stat = stat.set_index('protein')
    stat_d = stat['up/down']
    stat_p = stat['pvalue']
    stat_q = stat['qvalue']
    stat_f = stat['FC']
    
    if type(direction) == list:
        direction = stat_d
    else:
        direction = pd.concat([direction, stat_d], axis=1)
        
    if type(pvalue) == list:
        pvalue = stat_p
    else:
        pvalue = pd.concat([pvalue, stat_p], axis=1)
        
    if type(qvalue) == list:
        qvalue = stat_q
    else:
        qvalue = pd.concat([qvalue, stat_q], axis=1)
        
    if type(FC) == list:
        FC = stat_f
    else:
        FC = pd.concat([FC, stat_f], axis=1)
        
    count+=1
    print(count)
    
# direction.to_csv(os.path.join(working_dir, r'direction.txt'), sep='\t')
# pvalue.to_csv(os.path.join(working_dir, r'pvalue.txt'), sep='\t')
# qvalue.to_csv(os.path.join(working_dir, r'qvalue.txt'), sep='\t')
# FC.to_csv(os.path.join(working_dir, r'FC.txt'), sep='\t')



# stats up/down regulated ratio across resampling
direction_counts = direction.apply(count_elements_in_row, axis=1).fillna(0).astype(int)
direction_counts_percent = direction_counts.div(direction_counts.sum(axis=1), axis=0)
up_down_percent_pd = pd.concat([direction_counts_percent.idxmax(axis=1), direction_counts_percent.max(axis=1)], axis=1)
up_down_percent_pd.columns = ['Up/Down', 'consistent_ratio']


all_protein = list(proteome_train.columns)
all_protein.sort()


# cal fold change 99% CI
FC_CI = []
for each_protein in all_protein:

    fc_ = FC.loc[each_protein]
    up_, down_ = st.norm.interval(confidence=0.99, loc=np.mean(fc_), scale=st.sem(fc_))
    
    FC_CI.append([each_protein, np.mean(fc_), up_, down_])
FC_CI_pd = pd.DataFrame(FC_CI)
FC_CI_pd.columns = ['protein', 'FC', 'FC_CI-', 'FC_CI+']    
FC_CI_pd = FC_CI_pd.set_index('protein')



# cal adjusted pvalue 99% CI
qvalue_CI = []
for each_protein in all_protein:

    qvalue_ = qvalue.loc[each_protein]
    up_, down_ = st.norm.interval(confidence=0.99, loc=np.mean(qvalue_), scale=st.sem(qvalue_))
    
    qvalue_CI.append([each_protein, np.mean(qvalue_), up_, down_])
qvalue_CI_pd = pd.DataFrame(qvalue_CI)
qvalue_CI_pd.columns = ['protein', 'qvalue', 'qvalue_CI-', 'qvalue_CI+']    
qvalue_CI_pd = qvalue_CI_pd.set_index('protein')


# combine jackknifing results
jackniffing = pd.concat([proteome_train_isna_group, up_down_percent_pd, FC_CI_pd, qvalue_CI_pd], axis=1)



#%% filter up / down regulated protein biomarkers according jackknifing results
# cutoff
cutoff_qvalue = 0.05
cutoff_fc  = 1
cutoff_consistent_ratio = 1
cutoff_NA = 0.8



jackniffing_filter = jackniffing[(jackniffing['qvalue_CI+'] < cutoff_qvalue) & (jackniffing['FC_CI-'] > cutoff_fc)]
jackniffing_filter = jackniffing_filter[jackniffing_filter['consistent_ratio'] == cutoff_consistent_ratio]

ratio_select_filter = jackniffing[(jackniffing['FTB_ratio'] > cutoff_NA) | (jackniffing['PTB_ratio'] > cutoff_NA)]
ratio_select_filter_list = list(ratio_select_filter.index)

overlap_list = list(set(jackniffing_filter.index) & set(ratio_select_filter_list))
overlap_list.sort()


jackniffing_filter = jackniffing_filter.loc[overlap_list]

jackniffing_filter_up = jackniffing_filter[jackniffing_filter['Up/Down'] == 1]
jackniffing_filter_down = jackniffing_filter[jackniffing_filter['Up/Down'] == -1]



up_specific_mask = (jackniffing.index.isin(jackniffing_filter_up.index))
down_specific_mask = (jackniffing.index.isin(jackniffing_filter_down.index))

jackniffing['Legend'] = np.nan
jackniffing['Legend'][up_specific_mask] = 'up-specific'
jackniffing['Legend'][down_specific_mask] = 'down-specific'
jackniffing['Legend'][~(up_specific_mask | down_specific_mask)] = 'non-specific'

# stats sum of up/down biomarkers
up_specific_mask.sum()
down_specific_mask.sum()


#%% volcano plot


jackniffing['-log10(qvalue) CI+'] = -np.log10(np.abs(jackniffing['qvalue_CI+']))
jackniffing = jackniffing[~np.isinf(jackniffing['-log10(qvalue) CI+'])]
    

jackniffing['log2FoldChange CI-'] = np.log2(jackniffing['FC_CI-'])
jackniffing['log2FoldChange CI-'] = jackniffing['log2FoldChange CI-'] * jackniffing['Up/Down']


ax = sns.scatterplot(x="log2FoldChange CI-", y="-log10(qvalue) CI+", color='gray',  edgecolor=None, \
                      data=jackniffing, s=10, alpha=0.5, legend=False)
ax = sns.scatterplot(x="log2FoldChange CI-", y="-log10(qvalue) CI+",hue='Legend', \
                     hue_order = ('up-specific','non-specific', 'down-specific'),\
                  palette=("red","grey",'dodgerblue'),  edgecolor=None, \
                      data=jackniffing[(jackniffing['Legend'] == 'up-specific') | (jackniffing['Legend'] == 'down-specific')], s=20, alpha=0.5, legend=False)

plt.xlim([-6.5,6.5])
plt.ylim([-0.5,10])

plt.subplots_adjust(left=0.150, right=0.855, top=0.980 , bottom=0.14)

