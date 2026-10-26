# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 15:48:31 2023

@author: ZZG
"""


import pandas as pd
from kneed import KneeLocator
import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 18
plt.rcParams['font.family'] = 'arial'


trend = pd.read_excel(r'Extended Data Fig 6.xlsx', sheet_name='Extended Data Fig 6b', header=[0,1])

method = 'Average'

protein_number = list(trend.iloc[:,0])
median_list = list(trend.iloc[:,1])
max_list = list(trend.iloc[:,2])


kneed_median = KneeLocator(protein_number, median_list, S=1.0, curve="concave", direction="increasing", interp_method="polynomial", polynomial_degree=2)
kneed_median.plot_knee()


kneed_max = KneeLocator(protein_number, max_list, curve="concave", direction="increasing", interp_method="polynomial", polynomial_degree=2)
kneed_max.plot_knee()




