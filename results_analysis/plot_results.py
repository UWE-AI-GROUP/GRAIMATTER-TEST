import os
import numpy as np
import itertools
import random
import mcdm
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import gcf
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.pyplot import gca
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import scipy.stats as stats

#from data_preprocessing.data_interface import get_data_sklearn, DataNotAvailable

path = "/home/ec2-user/studies/GRAIMatter/experiments"
 
#path = os.path.join("","home", "ec2-user", "studies", "GRAIMatter", "experiments")
print(path)
file_names = [file for file in os.listdir(path) if os.path.isfile(os.path.join(path, file)) ]
#for files in os.listdir(path):
#    if os.path.isfile(os.path.join(path, files)):
#        print(files)
print(file_names)
#file_names = ["AdaBoost_results.csv",
#              "DecisionTree_results.csv", 
#              "Random_Forest_loop_results.csv",
#              "round_rf_results.csv",
#              "SVC_poly_results.csv",
#              "SVC_rbf_results.csv",
#              "SVC_rbf_dp_results.csv",
#              "xgboost_results.csv", 
#              "AdaBoost_results_minmax_round.csv",
#              "DecisionTreeClassifier_minmax_round_results.csv",
#              "round_minmax_rf_results.csv",
#              "round_rf_results.csv"
#             ]
results_df = pd.DataFrame()
for f in file_names:
    print(f)
    results_df = pd.concat([results_df, pd.read_csv(os.path.join(path, f))], ignore_index=True)
    #print(results_df.columns)
    #print(results_df.head())
results_df['target_classifier'] = [" ".join(x) for x in zip(list(results_df.target_classifier), list(results_df.kernel.fillna('')))]
print(results_df.head())
print(results_df.describe())
#print(results_df.columns)

print(results_df.dataset.unique())

results_df['dataset'].replace('minmax in-hospital-mortality', 'in-hospital-mortality', inplace=True)
results_df['dataset'].replace('minmax mimic2-iaccd', 'mimic2-iaccd', inplace=True)
results_df['dataset'].replace('minmax texas hospitals 10', 'texas hospitals 10', inplace=True)
results_df['dataset'].replace('minmax indian liver', 'indian liver', inplace=True)
results_df['dataset'].replace('minmax synth-ae', 'synth-ae', inplace=True)

#this chunk is when a problem was found with the metrics after running the experiments
tmp_min = results_df.mia_FMAX#wrong label, FMAX is in reality FMIN
results_df['mia_FMIN']=[round(x,8) for x in results_df.mia_FMAX]
results_df.mia_FMAX = [round(x,8) for x in tmp_min+results_df.mia_FDIF]
results_df.mia_FDIF = [round(x,8) for x in results_df.mia_FDIF]
del(tmp_min)
print(results_df.mia_PDIF.describe())
#IN FUTURE EXPERIMNETS THIS CAN BE DELETED###

print(len(results_df), results_df.target_classifier.unique())

common_vars = ['mia_TPR', 'mia_FPR', 'mia_FAR',
               'mia_TNR', 'mia_PPV', 'mia_NPV',
               'mia_FNR', 'mia_ACC', 'mia_F1score',
               'mia_Advantage', 'mia_AUC', 
               'mia_FMAX', 'mia_FMIN', 'mia_FDIF', 'mia_PDIF', 
               'mia_pred_prob_var']

xvars = ['mia_TPR', 'mia_FPR', 'mia_FAR',
       'mia_TNR', 'mia_PPV', 'mia_NPV', 'mia_FNR', 'mia_ACC', 'mia_F1score',
         'mia_FMAX', 'mia_FDIF', 'mia_PDIF', 
       'mia_Advantage', 'mia_AUC']#, 'mia_pred_prob_var']

yvars =  ['target_TPR', 'target_FPR', 'target_FAR', 
          'target_TNR', 'target_PPV',  'target_NPV', 'target_FNR', 'target_ACC', 'target_F1score',
          'target_FMAX', 'target_FDIF', 'target_PDIF', 
       'target_Advantage', 'target_AUC']#, 'target_pred_prob_var']

print(results_df['dataset'].unique)
print(results_df['target_classifier'].unique)

cmap = {'in-hospital-mortality':'b',
        'minmax in-hospital-mortality':'tab:blue',
        'round in-hospital-mortality':'tab:cyan',
        'indian liver':'g',
        'minmax indian liver':'tab:green',
        'round indian liver':'yellowgreen',
        'mimic2-iaccd':'r',
        'minmax mimic2-iaccd':'tab:red',
        'round mimic2-iaccd':'salmon',
        'synth-ae':'k',
        'minmax synth-ae':'tab:gray',
        'round synth-ae':'silver',
        'texas hospitals 10':'tab:orange',
        'minmax texas hospitals 10':'orange',
        'round texas hospitals 10':'goldenrod'
       }


datasets_features = {}
#for dataset in cmap.keys():
#    data_features, data_labels = get_data_sklearn(dataset)
#    datasets_features[dataset] = {'nrows':len(data_labels),
#                                  'ncols':len(results_df.columns),
#                                  'size':len(data_labels)*len(results_df.columns),
#                                  'n_binary_cols':sum([1 for x in data_features if len(data_features[x].unique)==2])
#                                 }

results_df['overfitting'] = results_df.target_train_ACC - results_df.target_ACC
results_df['overfitting AUC'] = results_df.target_train_AUC - results_df.target_AUC



#RISK SCORE OVERALL
cols = ['mia_TPR',
        'mia_FPR_reversed',
        'mia_TNR_reversed',
        'mia_PPV',
        'mia_NPV',
        'mia_FNR_reversed',
        'mia_ACC',
        'mia_F1score',
        'mia_FAR_reversed',
        'mia_Advantage',
        'mia_AUC',
        #'mia_FMAX',
        #'mia_FMIN_reversed',
        'mia_FDIF'
        ]
#mia_Advantage,mia_AUC,mia_FMAX,mia_FDIF,mia_PDIF

rev_cols = ["mia_FPR",
            "mia_FAR", 
            "mia_TNR",
            "mia_FNR", 
            #"mia_FMIN"
           ]
for var in rev_cols:
    results_df[var+"_reversed"] = [1-x for x in results_df[var]]

names = results_df.index #names of the experiments rows
mia_metrics_matrix = results_df[cols]   

#AUC 0.5 is optimal, adjust to this and normalise
minimum = 0.0
maximum = 0.5
mia_metrics_matrix.mia_AUC = [(abs(x-0.5)-minimum)/(maximum-minimum) for x in mia_metrics_matrix.mia_AUC]

#Normalise FDIF, as it has some negative values (it's a difference between fmax and fmin)
#zi = (xi – min(x)) / (max(x) – min(x))
mi = min(mia_metrics_matrix.mia_FDIF)
ma = max(mia_metrics_matrix.mia_FDIF)
mia_metrics_matrix.mia_FDIF = [(x-mi)/(ma-mi) for x in mia_metrics_matrix.mia_FDIF]


print(mia_metrics_matrix)
print(mia_metrics_matrix.describe())

print('min', min(mia_metrics_matrix.mia_FDIF), 'max', max(mia_metrics_matrix.mia_FDIF), 'FDIF')
#print('min', min(mia_metrics_matrix.mia_FMAX), 'max', max(mia_metrics_matrix.mia_FMAX), 'FMAX')
#print('min', min(mia_metrics_matrix.mia_FMIN_reversed), 'max', max(mia_metrics_matrix.mia_FMIN_reversed), 'FMIN')

score = mcdm.rank(mia_metrics_matrix, #is_benefit_x=is_benefit,
                  #w_method="SD",
                  #w_vector=[0.2, 0.4,0.4],
                  #w_vector=[0.05,
                  #          0.05,
                  #          0.2,
                  #          0.05,
                  #          0.05,
                  #          0.05,
                  #          0.05,
                  #          0.05,
                  #          0.05,
                  #          0.2,
                  #          0.2],
                alt_names=names,
                 # s_method="MEW"
                 )

score = sorted(score)

#print(max(list(zip(*score))[1]))
print(max(list(zip(*score))[1]))
print(min(list(zip(*score))[1]))
#print(score[-10:])
#print(list(zip(*mcdm.rank(mia_metrics_matrix, is_benefit_x=is_benefit, s_method="TOPSIS")))[1])


results_df['risk_score'] = list(zip(*score))[1]

#perform the Mann-Whitney U test
for clf in results_df.target_classifier.unique():
    print('Mann Whitney', clf)
    wc = results_df[(results_df.scenario=='WorstCase') &
                                       (results_df.target_classifier==clf)].risk_score
    s1 = results_df[(results_df.scenario=='Salem1') &
                                       (results_df.target_classifier==clf)].risk_score
    s2 = results_df[(results_df.scenario=='Salem2') &
                                       (results_df.target_classifier==clf)].risk_score
    print('worst case vs Salem1')
    print(stats.mannwhitneyu(wc, s1, alternative='two-sided'))
    print('worst case vs Salem2')
    print(stats.mannwhitneyu(wc, s2, alternative='two-sided'))


criteria = [results_df['risk_score']<=0.5, 
            results_df['risk_score'].between(0.5,0.7), results_df['risk_score']>0.7]
values = ['low', 'medium', 'high']
results_df['risk'] = np.select(criteria, values, 0)

#plotname = "boxplot_risk_scenario_classifier.png"
#if not os.path.exists(plotname+"X"):
#    print('doing', plotname)
#    g = sns.kdeplot(data=results_df, 
#                    x=v,  
#                    hue="risk")

plotname = "boxplot_risk_scenario_classifier.png"
if not os.path.exists(plotname):
    print('doing', plotname)
    g = sns.catplot(data=results_df,
                    x = 'target_classifier',
                    y = 'risk_score',
                    hue = 'scenario',
                    kind='box'
                   )
    g.set_xticklabels(rotation = 90)
    g.savefig(plotname)
    print(plotname,'done')
    plt.close()
    
plotname = "boxplot_risk_scenario_classifier_dataset.png"
if not os.path.exists(plotname):
    print('doing', plotname)
    g = sns.catplot(data=results_df,
                    x = 'target_classifier',
                    y = 'risk_score',
                    hue = 'scenario',
                    col = 'dataset',
                    kind='box'
                   )
    g.set_xticklabels(rotation = 90)
    g.savefig(plotname)
    print(plotname,'done')
    plt.close()


#Find risky hyperparms
plotname = "pointplot_riskscore_targetclassifer_param.pdf"
#if True: 
if not os.path.exists(plotname):
    print('doing', plotname)
    with PdfPages(plotname) as pdf_pages:
        i = 1
        for clf in results_df.target_classifier.unique():
            print(clf)
            tmp = results_df[results_df.target_classifier==clf]
            figu = plt.figure(i)
            g = sns.catplot(data=tmp,
                            x = 'param_id',
                            y = 'risk_score',
                            kind="point",
                            height=5, aspect=2,
                    )
            g.set_xticklabels(rotation = 90)
            plt.title(clf)
            pdf_pages.savefig(figu, bbox_inches='tight')
            i += 1
            del(tmp)
plt.close("all")
print(plotname, 'is done now')


nonparams = [#'dataset', 
             'scenario' ,
             #'target_classifier', 
             'shadow_classifier_name', 'shadow_dataset', 'attack_classifier', 'repetition', 'full_id' , 'model_data_param_id', 'param_id', 'target_TPR', 'target_FPR', 'target_FAR', 'target_TNR', 'target_PPV', 'target_NPV', 'target_FNR', 'target_ACC', 'target_F1score', 'target_Advantage', 'target_AUC', 'target_FMAX', 'target_FDIF', 'target_PDIF', 'target_pred_prob_var', 'target_train_TPR', 'target_train_FPR', 'target_train_FAR', 'target_train_TNR', 'target_train_PPV', 'target_train_NPV', 'target_train_FNR', 'target_train_ACC', 'target_train_F1score', 'target_train_Advantage', 'target_train_AUC', 'target_train_FMAX', 'target_train_FDIF', 'target_train_PDIF', 'target_train_pred_prob_var', 'mia_TPR', 'mia_FPR', 'mia_FAR', 'mia_TNR', 'mia_PPV', 'mia_NPV', 'mia_FNR', 'mia_ACC', 'mia_F1score', 'mia_Advantage', 'mia_AUC', 'mia_FMAX', 'mia_FMIN', 'mia_FDIF', 'mia_PDIF', 'mia_pred_prob_var', 'shadow_TPR', 'shadow_FPR', 'shadow_FAR', 'shadow_TNR', 'shadow_PPV', 'shadow_NPV', 'shadow_FNR', 'shadow_ACC', 'shadow_F1score', 'shadow_Advantage', 'shadow_AUC', 'shadow_FMAX', 'shadow_FDIF', 'shadow_PDIF', 'shadow_pred_prob_var', 'overfitting', 'overfitting AUC', 'mia_FAR_reversed', 'risk_score']


risky_hyperparams = results_df[results_df.risk_score>=0.7]
print('Number of risky experiments', len(risky_hyperparams))

param_id2values = {}
hyper_freq = {}
high_risk ={}
with open('risky_hyperparameters.csv','w') as out:
    out.write('target classifier\tdataset\thyperparameters\tfrequency\n')
    for pars in risky_hyperparams.items():
        for i,param_id in enumerate(risky_hyperparams.model_data_param_id.unique()):
            #param_id2values[param_id] = {}
            tmp = risky_hyperparams[risky_hyperparams.model_data_param_id == param_id]#.iloc[0]
            hyper_freq[param_id] = len(tmp)
            tmp = tmp.iloc[0]
            #print(tmp, tmp.loc['shadow_classifier_name'])#.isna())
            param_id2values[param_id] ={name:value for name, value in tmp.items() if name not in nonparams and str(value)!='nan' and name!='target_classifier' and name!='dataset'}# tmp.loc[name].notna()}
            target_classifier = tmp.target_classifier
            dataset = tmp.dataset
    #print(param_id2values)
            s = ''
            for item in param_id2values[param_id]:
                s += item + ':' + str(param_id2values[param_id][item]) + ' '
            high_risk[param_id] = {'target_classifier':target_classifier, 
                                  'dataset':dataset,
                                  'hyperparameters':s,
                                  'frequency': hyper_freq[param_id]}
            out.write(target_classifier + "\t" + dataset + "\t" + s +"\t" + str(hyper_freq[param_id]) + '\n')

df = pd.DataFrame.from_dict(high_risk, orient ='index')
#print(df)
#print(df[df.frequency>=5])
tmp = df.groupby(['target_classifier', 'hyperparameters']).sum()#['frequency'].sum()
print(tmp)
tmp = tmp.sort_values(['frequency', 'target_classifier','hyperparameters'], ascending=False)
#print(sorted(tmp[tmp['frequency']>5]))
print('tmp sorted')
print(tmp)
tmp.to_csv('hihg_risk_hyperparams_morethan5obersvations.csv')


#print(hyper_freq)
print('hyper risky done')

#{hyper_freq[param_id]= for param_id in risky_hyperparams.model_data_param_id.unique()}
        


plotname = "scatterplot_risk_scoreVSmia_AUC_FMAX_by_scenario_dataset.png"
#if True: 
if not os.path.exists(plotname):
    g=sns.relplot(data=results_df,
            y='risk_score',
            x='mia_AUC',
            hue="mia_FMAX",
            col="scenario",
            row='dataset',
            kind='scatter')#,
            #cut=0)
    #plt.axhspan(0.45, 0.55, color='orange', alpha=0.25, lw=0)
    g.savefig(plotname)
    plt.close()
    
plotname = "scatterplot_risk_scoreVSmia_AUC_by_dataset.png"
#if True: 
if not os.path.exists(plotname):
    g=sns.relplot(data=results_df,
            y='risk_score',
            x='mia_AUC',
            hue='dataset',
            kind='scatter')#,
            #cut=0)
    #plt.axhspan(0.45, 0.55, color='orange', alpha=0.25, lw=0)
    g.savefig(plotname)
    plt.close()

plotname = "scatterplot_risk_scoreVSmia_FAR_by_dataset.png"
#if True: 
if not os.path.exists(plotname):
    g=sns.relplot(data=results_df,
            y='risk_score',
            x='mia_FAR',
            hue='dataset',
            kind='scatter')#,
            #cut=0)
    #plt.axhspan(0.45, 0.55, color='orange', alpha=0.25, lw=0)
    g.savefig(plotname)
    plt.close()

plotname = "scatterplot_risk_scoreVSmia_Advantage_by_dataset.png"
#if True: 
if not os.path.exists(plotname):
    g=sns.relplot(data=results_df,
            y='risk_score',
            x='mia_Advantage',
            hue='dataset',
            kind='scatter')#,
            #cut=0)
    #plt.axhspan(0.45, 0.55, color='orange', alpha=0.25, lw=0)
    g.savefig(plotname)
    plt.close()

plotname = "scatterplot_risk_scoreVSmia_FDIF_by_dataset.png"
#if True: 
if not os.path.exists(plotname):
    g=sns.relplot(data=results_df,
            y='risk_score',
            x='mia_FDIF',
            hue='dataset',
            kind='scatter')#,
            #cut=0)
    #plt.axhspan(0.45, 0.55, color='orange', alpha=0.25, lw=0)
    g.savefig(plotname)
    plt.close()

plotname = "scatterplot_risk_scoreVSmia_AUC_FDIF_by_scenario_dataset.png"
#if True: 
if not os.path.exists(plotname):
    g=sns.relplot(data=results_df,
            y='risk_score',
            x='mia_AUC',
            hue="mia_FDIF",
            col="scenario",
            row='dataset',
            kind='scatter')#,
            #cut=0)
    plt.axhspan(0.45, 0.55, color='orange', alpha=0.25, lw=0)
    g.savefig(plotname)
    plt.close()

plotname = "joint_scatterplot_RandomForest_riskscore_WorstCase_overfitting_minsamplesleaf.png"
#if True: 
if not os.path.exists(plotname):
    print(plotname)
    tmp = results_df[(results_df['target_classifier']=='RandomForestClassifier ') &
                            (results_df['scenario']=='WorstCase')]
    g = sns.jointplot(data=tmp,
                      x='overfitting', 
                      y = 'risk_score',
                      hue ='min_samples_leaf',
                      kind="kde",
                            )
    #g.refline(y=0.7, x=0.25)
    plt.tight_layout()
    g.savefig(plotname)
    plt.close()

plt.close("all")
    
plotname = "jointplot_riskscore_WorstCase_overfitting.pdf"
#if True: 
if not os.path.exists(plotname):
    with PdfPages(plotname) as pdf_pages:
        i = 1
        for dataset in results_df.dataset.unique():
            for scenario in results_df.scenario.unique():
                figu = plt.figure(i)
                print(dataset)
                tmp = results_df[(results_df['dataset']==dataset) &
                                    (results_df['scenario']==scenario)]
                g = sns.jointplot(data=tmp,
                          x='overfitting', 
                          y = 'risk_score',
                          hue ='target_classifier',
                          kind="kde",
                                )
                plt.title(dataset+" - "+scenario)
                #g.set(title=dataset)
                g.refline(y=0.6, x=0.25)
                #plt.tight_layout()
                #plt.tight_layout()
                pdf_pages.savefig(figu, bbox_inches='tight')
                i+=1
    plt.close("all")
print(plotname)

plt.close("all")
plotname = "joint_scatterplot_riskscore_WorstCase_overfitting_byclassifier.pdf"
#if True:
if not os.path.exists(plotname):
    print('doing ',plotname)
    with PdfPages(plotname) as pdf_pages:
        i = 1
        for clf in results_df.target_classifier.unique():
            print(clf)
            for scenario in results_df.scenario.unique():
                figu = plt.figure(i)
                tmp = results_df[(results_df['target_classifier']==clf) &
                                    (results_df['scenario']==scenario)]
                g = sns.jointplot(data=tmp,
                          x='overfitting', 
                          y = 'risk_score',
                          hue ='dataset',
                          kind="kde",
                                )#.set_ylabel('risk - '+clf )
                plt.title(clf + "-" + scenario)
                #g.set(title=dataset)
                g.refline(y=0.6, x=0.25)
                #plt.tight_layout()
                #plt.tight_layout()
                pdf_pages.savefig(figu, bbox_inches='tight')
                i+=1
    plt.close("all")
print(plotname)
    #g.savefig(plotname)

plotname = "joint_kdeplot_riskscore_WorstCase_overfitting_RandomForest.pdf"
#if True:
if not os.path.exists(plotname):
    print('doing ',plotname)
    tmp = results_df[(results_df['target_classifier']=='RandomForestClassifier ') &
                                (results_df['scenario']=='WorstCase')]
    tmp['Dataset - leaves'] = tmp.dataset + '-' + tmp.min_samples_leaf.astype('str')
    g = sns.jointplot(data=tmp,
                      x='overfitting', 
                      y = 'risk_score',
                      hue ='Dataset - leaves',
                      kind="kde",
                            )
    #g.set(title=dataset)
    g.refline(y=0.6, x=0.25)
    #plt.tight_layout()
    #plt.tight_layout()
    plt.legend(bbox_to_anchor=(1.02, 1), loc=2)
    g.savefig(plotname)
    plt.close("all")
print(plotname)
    
plotname = "pointplot_Classifier_riskscore_scenario_by_dataset.png"
#if True:
if not os.path.exists(plotname):
    print(plotname)    
    #tmp = results_df[#(results_df['target_classifier']=='DecisionTreeClassifier') &
    #                 (results_df['scenario']=='WorstCase')]
    g = sns.catplot(data=results_df,
                    y = 'risk_score',
                    hue ='dataset',
                    x='target_classifier',
                    col='scenario',
                    kind="point",
                    height=5, aspect=1.1,
            )
    g.set_xticklabels(rotation = 90)
    g.savefig(plotname)
    plt.close()

plotname = "pointplot_RandomForest_riskscore_minsamplesleaf_by_dataset.png"
if False:
#if True:
#if not os.path.exists(plotname):
    print(plotname)    
    tmp = results_df[(results_df['target_classifier']=='RandomForestClassifier') &
                     (results_df['scenario']=='WorstCase')]
    g = sns.catplot(data=tmp,
                    x='min_samples_leaf', 
                    y = 'risk_score',
                    hue ='dataset',
                    #col='target_classifier',
                    kind="point",
                    height=5, aspect=1.1,
            )
    g.savefig(plotname)
    plt.close()

plotname = "pointplot_SVC-rbf_riskscore_gamma_by_dataset.png"
if not os.path.exists(plotname):
    print(plotname)    
    tmp = results_df[(results_df['target_classifier']=='SVC rbf') &
                    (results_df['gamma']!='scale') & (results_df['scenario']=='WorstCase')]
    tmp.gamma = tmp.gamma.astype(float)
    tmp = tmp[tmp.gamma<100.0]
    g = sns.catplot(data=tmp,
                    x='gamma', 
                    y = 'risk_score',
                    hue ='dataset',
                    col='target_classifier',
                    kind="point",
                    height=5, aspect=1.1,
            )
    g.savefig(plotname)
    plt.close()

plotname = "joint_scatterplot_RandomForest_miaAUC_WorstCase_overfitting_by_minsamplesleaf.png"
if not os.path.exists(plotname):
    print(plotname)
    tmp = results_df[(results_df['target_classifier']=='RandomForestClassifier ') &
                            (results_df['scenario']=='WorstCase')]
    g = sns.jointplot(data=tmp,
                      x='overfitting', 
                      y = 'mia_AUC',
                      hue ='min_samples_leaf',
                      kind="kde",
                            )
    g.refline(y=0.7, x=0.25)
    plt.tight_layout()
    g.savefig(plotname)
    plt.close()
    
plotname = "jointplot_SVC-rbf_miaAUC_gamma_by_dataset.png"    
if not os.path.exists(plotname):
    print(plotname)
    tmp = results_df[(results_df['target_classifier']=='RandomForestClassifier ') &
                            (results_df['scenario']=='WorstCase')]
    g = sns.jointplot(data=tmp,
                      x='overfitting', 
                      y = 'risk_score',
                      hue ='gamma',
                      kind="kde",
                            )
    g.refline(y=0.6, x=0.25)
    plt.title('Random Forest classsifier - Worst case scenario')
    #plt.tight_layout()
    g.savefig(plotname)
    plt.close()

plotname = "pointplot_SVC-rbf_miaAUC_gamma_by_dataset.png"
if not os.path.exists(plotname):
    print(plotname)    
    tmp = results_df[(results_df['target_classifier']=='SVC rbf') &
                    (results_df['gamma']!='scale') & (results_df['scenario']=='WorstCase')]
    tmp.gamma = tmp.gamma.astype(float)
    tmp = tmp[tmp.gamma<100.0]
    g = sns.catplot(data=tmp,
                    x='gamma', 
                    y = 'mia_AUC',
                    hue ='dataset',
                    col='target_classifier',
                    kind="point",
                    height=5, aspect=1.1,
            )
    g.savefig(plotname)
    plt.close()

plotname = "joint_kdeplot_miaAUC_vs_overfitting_by_scenario.png"
if not os.path.exists(plotname):
    print(plotname)
    sns.set_style("whitegrid")
    g = sns.jointplot(data=results_df,
                  x='overfitting', 
                  y = 'mia_AUC',
                  hue ='scenario',
                  kind="kde"
                        )
    #g.fig.suptitle("")
    plt.tight_layout()
    g.savefig(plotname)
    plt.close()
    
plotname = "joint_kdeplot_miaAUC_vs_overfitting_by_dataset.png"
if not os.path.exists(plotname):
    print(plotname)
    sns.set_style("whitegrid")
    g = sns.jointplot(data=results_df,
                  x='overfitting', 
                  y = 'mia_AUC',
                  hue ='dataset',
                  kind="kde"
                        )
    #g.fig.suptitle("")
    plt.tight_layout()
    g.savefig(plotname)
    plt.close()
    
plotname = "joint_kdeplot_miaAUC_vs_overfittingAUC_by_scenario.png"
if False: #not os.path.exists(plotname):
    print(plotname)
    sns.set_style("whitegrid")
    g = sns.jointplot(data=results_df,
                  x='overfitting AUC', 
                  y = 'mia_AUC',
                  hue ='scenario',
                  kind="kde"
                        )
    #g.fig.suptitle("")
    plt.tight_layout()
    g.savefig(plotname)
    plt.close()
    
    
#plotname = "joint_kdeplot_miaAUC_WC-Salem1_by_classifier.png"
#if not os.path.exists(plotname):
#    print(plotname)
sns.set_style("whitegrid")   
mia_metrics = [column for column in results_df.columns if "mia_" in column]
other_columns = [column for column in results_df.columns if "mia_" not in column]
cols = ['target_classifier', 'repetition', 
                    'dataset', 'param_id','model_data_param_id'
                   ]
worstcase = results_df[results_df.scenario=="WorstCase"].set_index(cols)[mia_metrics]
salem1 = results_df[results_df.scenario=="Salem1"].set_index(cols)[mia_metrics]
salem2 = results_df[results_df.scenario=="Salem2"].set_index(cols)[mia_metrics]

wc_s1 = worstcase-salem1
wc_s1.reset_index(inplace=True)
wc_s1.drop("repetition", axis=1, inplace=True)
wc_s1 = wc_s1.groupby(['target_classifier', 'dataset', 'param_id']).mean().reset_index()


plotname = 'pointplots_classifier_dataset_scenario.pdf'
if not os.path.exists(plotname):
    print(plotname)
    sns.set_style("whitegrid")
    sns.set_palette(cmap.values())
    #mcolors.get_named_colors_mapping().update(cmap)
    with PdfPages(plotname) as pdf_pages:
        i = 1
        for v in common_vars:
            print(v)
            figu = plt.figure(i)
            g = sns.catplot(data=results_df,
                        x="target_classifier",
                        y=v,
                        hue="dataset", #hue_order=cmap.keys(),
                        #row=,
                        col="scenario",
                        kind="point",
                        height=5, aspect=0.8,
            )
            if v!='mia_PDIF':
                g.set(ylim=(-0.05, 1.05))
            #g.set_title(v)
            #g.set(ylim=(-0.05, 1.05))
            g.set_xticklabels(rotation = 90)
            #plt.xticks(rotation=90)
            #plt.show()
            #plt.tight_layout()
            pdf_pages.savefig(figu, bbox_inches='tight')
            i+=1
    plt.close("all")

plotname = 'pairplots_metrics.pdf'
if not os.path.exists(plotname):
    print(plotname)
    sns.set_palette("Set1")
    with PdfPages(plotname) as pdf_pages:
        i = 1
        for target_classifier in results_df.target_classifier.unique():
            tmp = results_df[(results_df.target_classifier==target_classifier) &
                            (results_df.scenario=="WorstCase")]
            if len(tmp)>500:
                #for clf in tmp.target_classifier.unique():
                figu = plt.figure(i) 
                g = sns.pairplot(tmp, 
                                 x_vars = xvars,
                                 y_vars = xvars,
                                 hue ='dataset',
                                 corner=True,
                                 kind = 'reg')#reg
                #g.set_title(target_classifier)
                plt.title(target_classifier)
                #g.fig.suptitle(dataset + ' ' + clf)
                #plt.tight_layout()
                pdf_pages.savefig(figu, bbox_inches='tight')
                i+=1
    plt.close("all")

plotname = 'pairplots_metrics_dataset_classifier_scenario.pdf'
if not os.path.exists(plotname):
    print(plotname)
    sns.set_palette("Set1")
    with PdfPages(plotname) as pdf_pages:
        i = 1
        for dataset in set(list(results_df.dataset)):
            for clf in set(list(results_df.target_classifier)):
                tmp = results_df[(results_df['dataset']==dataset) & 
                                            (results_df['target_classifier']==clf)]
                if len(tmp)>500:
                    print(dataset, clf)
                    figu = plt.figure(i) 
                    g = sns.pairplot(tmp, 
                                     x_vars = xvars,
                                     y_vars = xvars,
                                     hue ='scenario',
                                     corner=True,
                                 kind = 'scatter')#reg
                    #g.set_title(dataset+' '+clf)
                    g.fig.suptitle(dataset + ' ' + clf)
                    #plt.tight_layout()
                    pdf_pages.savefig(figu, bbox_inches='tight')
                    i+=1
    plt.close("all")


plotname = 'violinplots_metrics_by_metrics_classifier.pdf'
if not os.path.exists(plotname):
    print(plotname)
    with PdfPages(plotname) as pdf_pages:
        i = 0
        for v in common_vars:
            if not True in pd.isna(results_df[v].unique):    
                figu = plt.figure(i)
                g = sns.catplot(data=results_df,
                            x="target_classifier", y=v,
                            col="scenario",
                            row="dataset",
                            kind="violin", cut=0 ,
                            inner="quartile",
                            height=3, aspect=2
                )
                #g.set_title(v)
                plt.xticks(rotation=90)
                plt.tight_layout()
                pdf_pages.savefig(figu)
                i+=1
    plt.close("all")


######################
# compare scenarios  #
######################
print('Compare scenarios')

type_datasets = ['not_normalised', 'round', 'minmax']
if True:
    for target_clf in results_df.target_classifier.unique():
        #tmp_df = results_df[results_df.target_classifier==target_clf]
        #print(len(tmp_df), target_clf)
            df = results_df[results_df.target_classifier==target_clf]
            t_dataset = ''
        #for t_dataset in type_datasets:
            #print(t_dataset)
            #if t_dataset != 'not_normalised':
            #    df = tmp_df[tmp_df.dataset.isin([x for x in tmp_df.dataset.unique() if t_dataset in x])]
            #else:
            #    df = tmp_df[tmp_df.dataset.isin([x for x in tmp_df.dataset.unique() if 'round' not in x and 'minmax' not in x])]
            #print('len df after selecting dataset', len(df))
            if len(df)>0:
                mia_metrics = [column for column in df.columns if "mia_" in column]
                mia_metrics = ['mia_TPR', 'mia_FPR', 'mia_FAR',
                   'mia_TNR', 'mia_PPV', 'mia_NPV',
                   'mia_FNR', 'mia_ACC', 'mia_F1score',
                   'mia_Advantage', 'mia_AUC', 
                   #'mia_FMAX',
                               'mia_FDIF',
                               'risk_score',#'mia_PDIF', 
                   #'mia_pred_prob_var'
                              ]
                other_columns = [column for column in df.columns if "mia_" not in column]
                cols = ['target_classifier', 'repetition', 
                        'dataset', 'param_id','model_data_param_id'
                       ]

                worstcase = df[df.scenario=="WorstCase"].set_index(cols)[mia_metrics]
                salem1 = df[df.scenario=="Salem1"].set_index(cols)[mia_metrics]
                salem2 = df[df.scenario=="Salem2"].set_index(cols)[mia_metrics]

                wc_s1 = worstcase-salem1
                wc_s1.reset_index(inplace=True)
                wc_s1.drop("repetition", axis=1, inplace=True)
                wc_s1 = wc_s1.groupby(['target_classifier', 'dataset', 'param_id']).mean().reset_index()

                target_classifier = wc_s1.pop("target_classifier")
                dataset = wc_s1.pop("dataset")
                param = wc_s1.pop("param_id")

                clf = dict(zip(set(target_classifier), sns.color_palette("Set1")))
                dt = dict(zip(set(dataset), sns.color_palette("tab20")))
                #p = dict(zip(set(param), sns.color_palette(a)))

                tc_colors = [clf[c] for c in target_classifier]#target_classifier.map(clf)
                #par_col = [p[c] for c in param]#param.map(p)
                data_colors = [cmap[c] for c in dataset]#list(dataset.map(dt))

                row_colors = pd.DataFrame({#'Target classifier':tc_colors,
                                          'Dataset':data_colors,
                                          #'parameters':par_col
                                          })

                custom_params = {"figure.subplot.right": 0.3}
                sns.set_theme(rc=custom_params)
                #fig, ax1 = plt.subplots(1, 1, figsize=(15,30))
                g = sns.clustermap(wc_s1,  cmap='seismic',
                                   vmin=-1.0, vmax=1.0,
                                   figsize=(15, 30),
                                   row_colors=row_colors,
                                   col_cluster=False,
                                   dendrogram_ratio=(.1, .1),
                                   yticklabels=False,
                                   cbar_pos=(0.99, .65, .03, .25),
                                   #ax=ax1
                                  )
                plt.title(target_clf+" Worst case - Salem1", x=-25, fontsize=15)

                d_legend_lines = [Line2D([0], [0], color=colour, lw=7) for colour in cmap.values()]
                d_legend_names = [label for label in cmap.keys()]

                #c_legend_lines = [Line2D([0], [0], color=colour, lw=7) for colour in clf.values()]
                #c_legend_names = [label for label in clf.keys()]

                l1 = plt.legend(d_legend_lines, d_legend_names,
                                title='Dataset', 
                                loc='lower left', bbox_to_anchor=(0, -1.2))
                #l2 = plt.legend(c_legend_lines, c_legend_names,
                #                title='Target classifier', 
                #                loc='lower left', bbox_to_anchor=(0, -2.0))
                #gca().add_artist(l2)
                gca().add_artist(l1)
                plt.savefig(target_clf+"_"+t_dataset+"_Clustermap_WorstCase_minus_Salem1.pdf", bbox_inches='tight')
                plt.close()


                wc_s2 = worstcase-salem2
                #print(wc_s1)
                wc_s2.reset_index(inplace=True)
                wc_s2.drop("repetition", axis=1, inplace=True)
                wc_s2 = wc_s2.groupby(['target_classifier', 'dataset', 'param_id']).mean().reset_index()
                target_classifier = wc_s2.pop("target_classifier")#list(wc_s1.index.get_level_values(0))#
                #print(target_classifier)
                dataset = wc_s2.pop("dataset")#list(wc_s1.index.get_level_values(1))#'dataset')
                #print(dataset)
                param = wc_s2.pop("param_id")#list(wc_s1.index.get_level_values(2))#'param_id')#
                clf = dict(zip(set(target_classifier), sns.color_palette("Set1")))
                dt = dict(zip(set(dataset), sns.color_palette("Paired")))
                print(dt)
                #p = dict(zip(set(param), sns.color_palette(a)))
                tc_colors = [clf[c] for c in target_classifier]#target_classifier.map(clf)
                #par_col = [p[c] for c in param]#param.map(p)
                #data_colors = [dt[c] for c in dataset]#list(dataset.map(dt))
                data_colors = [cmap[c] for c in dataset]
                #print(data_colors)
                #sns.set_palette("viridis")
                row_colors = pd.DataFrame({#'Target classifier':tc_colors,
                                          'Dataset':data_colors,
                                          #'parameters':par_col
                                          })
                #fig, ax1 = plt.subplots(1, 1, figsize=(20,20))
                g = sns.clustermap(wc_s2,  cmap='seismic',
                                   vmin=-1.0, vmax=1.0,
                                   row_colors=row_colors,
                                   figsize=(15, 20),
                                   col_cluster=False,
                                   dendrogram_ratio=(.1, .1),
                                   yticklabels=False,
                                   cbar_pos=(0.99, .65, .03, .25),
                                   #ax=ax1
                                  )
                plt.title(target_clf+" Worst case - Salem2", x=-25, fontsize=15)


                d_legend_lines = [Line2D([0], [0], color=colour, lw=7) for colour in cmap.values()]
                d_legend_names = [label for label in cmap.keys()]

                #c_legend_lines = [Line2D([0], [0], color=colour, lw=7) for colour in clf.values()]
                #c_legend_names = [label for label in clf.keys()]

                l1 = plt.legend(d_legend_lines, d_legend_names,
                                title='Dataset', 
                                loc='lower left', bbox_to_anchor=(0, -1.2))
                #l2 = plt.legend(c_legend_lines, c_legend_names,
                #                title='Target classifier', 
                #                loc='lower left', bbox_to_anchor=(0, -2.0))
                #gca().add_artist(l2)
                gca().add_artist(l1)

                plt.savefig(target_clf+"_"+t_dataset+"_Clustermap_WorstCase_minus_Salem2.pdf", bbox_inches='tight')
                plt.close()
            
            
#########
# 
#########

for f in dataset_features:
#datasets_features
    sns.relplot(
        data=results_df, 
        x="target_AUC", y="target_train_AUC",
        hue=f, 
        style="scenario", 
        col="dataset",
        row="target_classifier",
        kind="scatter", alpha=0.3
    )
    plt.savefig(f+"_scatterplot_AUC_target_test_train_by_dataset_feature.pdf", bbox_inches='tight')
    plt.close()