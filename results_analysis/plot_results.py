import os
import numpy as np
import itertools
import random
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import gcf
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.pyplot import gca
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
#from data_preprocessing.data_interface import get_data_sklearn, DataNotAvailable

path = "~/studies/GRAIMatter/experiments"
file_names = ["AdaBoost_results.csv",
              "DecisionTree_results.csv", 
              "Random_Forest_loop_results.csv",
              "round_rf_results.csv",
              "SVC_poly_results.csv",
              "SVC_rbf_results.csv",
              "SVC_rbf_dp_results.csv",
              "xgboost_results.csv", 
              "AdaBoost_results_minmax_round.csv",
              "DecisionTreeClassifier_minmax_round_results.csv",
              "round_minmax_rf_results.csv",
              "round_rf_results.csv"
             ]
results_df = pd.DataFrame()
for f in file_names:
    results_df = pd.concat([results_df, pd.read_csv(os.path.join(path, f))], ignore_index=True)
    
results_df['target_classifier'] = [" ".join(x) for x in zip(list(results_df.target_classifier), list(results_df.kernel.fillna('')))]

print(len(results_df), results_df.target_classifier.unique())

common_vars = ['mia_TPR', 'mia_FPR', 'mia_FAR',
               'mia_TNR', 'mia_PPV', 'mia_NPV',
               'mia_FNR', 'mia_ACC', 'mia_F1score',
               'mia_Advantage', 'mia_AUC', 'mia_pred_prob_var']

xvars = ['mia_TPR', 'mia_FPR', 'mia_FAR',
       'mia_TNR', 'mia_PPV', 'mia_NPV', 'mia_FNR', 'mia_ACC', 'mia_F1score',
       'mia_Advantage', 'mia_AUC']#, 'mia_pred_prob_var']

yvars =  ['target_TPR', 'target_FPR', 'target_FAR', 
          'target_TNR', 'target_PPV',  'target_NPV', 'target_FNR', 'target_ACC', 'target_F1score',
       'target_Advantage', 'target_AUC', 'target_pred_prob_var']

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
g.savefig("joint_scatterplot_RandomForest_miaAUC_WorstCase_overfitting_by_minsamplesleaf.png")

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
g.savefig("pointplot_SVC-rbf_miaAUC_gamma_by_dataset.png")

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
    
plotname = "joint_kdeplot_miaAUC_vs_overfittingAUC_by_scenario.png"
if not os.path.exists(plotname):
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
    
    
plotname = "joint_kdeplot_miaAUC_WC-Salem1_by_classifier.png"
if not os.path.exists(plotname):
    print(plotname)
    sns.set_style("whitegrid")   
    mia_metrics = [column for column in df.columns if "mia_" in column]
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


plotname = 'pointplots_classifier_dataset_scenario.pdf'
if not os.path.exists(plotname):
    print(plotname)
    sns.set_style("whitegrid")
    sns.set_palette(cmap.values())
    #mcolors.get_named_colors_mapping().update(cmap)
    with PdfPages(plotname) as pdf_pages:
        i = 0
        for v in common_vars:
            figu = plt.figure(i)        
            g = sns.catplot(data=results_df,
                        x="target_classifier",
                        y=v,
                        hue="dataset", hue_order=cmap.keys(),
                        #row=,
                        col="scenario",
                        kind="point",
                        height=5, aspect=0.8,
            ).set(ylim=(-0.05, 1.05))
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
        #i = 1
        #for dataset in set(list(results_df.dataset)):
        #    for clf in set(list(results_df.target_classifier)):
        figu = plt.figure(0) 
        g = sns.pairplot(results_df, #[(results_df['dataset']==dataset) & 
                                     #       (results_df['target_classifier']==clf)], 
                                 x_vars = xvars,
                                 y_vars = xvars,
                                 hue ='scenario',
                             corner=True,
                             kind = 'scatter')#reg
                #g.set_title(dataset+' '+clf)
                #g.fig.suptitle(dataset + ' ' + clf)
        plt.tight_layout()
        pdf_pages.savefig(figu, bbox_inches='tight')
        #i+=1
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