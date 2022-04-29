'''
apply multiple-criteria decision-making for scoring experiments.
Low score should represent safer models, and hihgher scores are more risky scores.

It aggregates all the MIA metrics calculated in the experiments.
'''

import mcdm
import pandas as pd

results_filename = "/home/ec2-user/studies/GRAIMatter/experiments/SVC_poly_results.csv"#Random_Forest_loop_results.csv"
results_df = pd.read_csv(results_filename)
print(results_df.columns)
print(results_df.head())

cols = ['mia_TPR', 'mia_FPR', 
        'mia_FAR','mia_TNR', 
        'mia_PPV', 'mia_NPV',
        'mia_FNR', 'mia_ACC',
        'mia_F1score', 'mia_Advantage',
        'mia_AUC']

#list defining whether the cols are benefit(True) or cost(False)
#understanding as benefit low risk of attack and cost as high risk of attack
# e.g. high values of mia_AUC means the attacker perform better, set benefit to False
is_benefit = [False, True,
             True, True,
             False, True,
             True, False,
             False, False,
             False]

names = results_df.index #names of the experiments rows
mia_metrics_matrix = results_df[cols]   

score = mcdm.rank(mia_metrics_matrix, is_benefit_x=is_benefit,
                       alt_names=names,
                       s_method="mTOPSIS")

#print(sorted(score))
#print(max(list(zip(*score))[1]))
print(max(list(zip(*score))[1]))
print(min(list(zip(*score))[1]))
#print(list(zip(*mcdm.rank(mia_metrics_matrix, is_benefit_x=is_benefit, s_method="TOPSIS")))[1])
