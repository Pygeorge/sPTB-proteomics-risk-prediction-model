# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 11:38:25 2024

@author: ZZG
"""


def composite_score(combine_pd):

    combine_pd['qvalue_rank'] = combine_pd['qvalue_CI+'].rank(method='dense', ascending=False, pct = True)
    combine_pd['FC_rank'] = combine_pd['FC_CI-'].rank(method='dense', ascending=True, pct = True)
    combine_pd['AUC_rank'] = combine_pd['AUC'].rank(method='dense', ascending=True, pct = True)
    combine_pd['support_rank'] = combine_pd['Counter'].rank(method='dense', ascending=True, pct = True)
    combine_pd['comprehensive_rank'] = combine_pd[['qvalue_rank', 'FC_rank', 'AUC_rank', 'support_rank']].mean(axis=1)
    combine_pd = combine_pd.sort_values(by='comprehensive_rank', ascending=False)
    
    return combine_pd