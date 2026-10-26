# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 11:10:49 2022

@author: zhaozhiguang1
"""




import umap
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os



plt.rcParams['font.size'] = 20
plt.rcParams['font.family'] = 'arial'

#%%
# input data
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



protein_filter = pd.read_excel(r'..\table\Source Data Fig. 2.xlsx', sheet_name='Figure2d', index_col=0)
protein_filter_list = list(protein_filter.index)

overlap_protein = list(set(protein_filter_list) & set(proteome_train.columns))
len(overlap_protein)

input_clinic_info = clinic_info
input_protein_select = proteome_train[overlap_protein]
input_protein = np.log2(input_protein_select)

#%% UMAP

region_order = ['FTB', 'PTB']

# UMAP parameters
i = 10
j = 0.1


# run UMAP fit
mapper = umap.UMAP(n_neighbors=i,min_dist=j, n_components=2, metric='euclidean', random_state=123).fit(input_protein.values)
u = mapper.fit_transform(input_protein)


# plot UMAP scatter
u_pd= pd.DataFrame(u)
u_pd.columns = ['UMAP (x)','UMAP (y)']
u_pd.index = input_protein.index.values


u_pd['flag'] = [input_clinic_info.loc[i]['Group'] for i in u_pd.index]


u_pd = u_pd.sort_values(by='flag')
        
fig = plt.figure(figsize=(8,6))
ax = sns.scatterplot(x="UMAP (x)", y="UMAP (y)", hue="flag",  data=u_pd, s=100, edgecolor='k', hue_order=region_order, 
                palette=['#FDC7C8', '#FF66FF'])

plt.legend(frameon=False, bbox_to_anchor=[1,1,0,0])
plt.subplots_adjust(left=0.140, right=0.780, top=0.910 , bottom=0.125)

plt.title( 'n-neighbors:' + str(int(i)) + '  dist:' + str(round(j, 1)), pad=10)

plt.savefig( 'n-neighbors_' + str(int(i)) + ' dist_' +  str(round(j, 1)) + '.png')
plt.close()




