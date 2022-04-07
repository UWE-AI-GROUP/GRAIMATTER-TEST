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
        'minmax texas hospitals 10':'orange'
       }

sns.set_style("whitegrid")
sns.set_palette(cmap.values())
#mcolors.get_named_colors_mapping().update(cmap)
with PdfPages('pointplots_classifier_dataset_scenario.pdf') as pdf_pages:
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
        plt.tight_layout()
        pdf_pages.savefig(figu)
        i+=1
        
sns.set_palette("Set1")
with PdfPages('pairplots_metrics.pdf') as pdf_pages:
    i = 1
    #for dataset in set(list(results_df.dataset)):
    #    for clf in set(list(results_df.target_classifier)):
    figu = plt.figure(i) 
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
    pdf_pages.savefig(figu)
    i+=1
    
sns.set_palette("Set1")
with PdfPages('pairplots_metrics_dataset_classifier_scenario.pdf') as pdf_pages:
    i = 1
    for dataset in set(list(results_df.dataset)):
        for clf in set(list(results_df.target_classifier)):
            tmp = results_df[(results_df['dataset']==dataset) & 
                                        (results_df['target_classifier']==clf)]
            if len(tmp)>100:
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
                plt.tight_layout()
                pdf_pages.savefig(figu)
                i+=1

with PdfPages('violinplots_metrics_by classifier.pdf') as pdf_pages:
    i = 0
    for v in common_vars:
        figu = plt.figure(i)        
        g = sns.catplot(data=results_df,
                    row="target_classifier", y=v,
                    x="scenario",
                    #row=,
                    hue="dataset",
                    kind="violin", cut=0 ,
                    inner="quartile",
                    height=3, aspect=2
        )
        #g.set_title(v)
        plt.xticks(rotation=90)
        plt.show()
        plt.tight_layout()
        pdf_pages.savefig(figu)
        i+=1


        
######################
# compare scenarios  #
######################
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

target_classifier = wc_s1.pop("target_classifier")
dataset = wc_s1.pop("dataset")
param = wc_s1.pop("param_id")

clf = dict(zip(set(target_classifier), sns.color_palette("Set1")))
dt = dict(zip(set(dataset), sns.color_palette("Paired")))
#p = dict(zip(set(param), sns.color_palette(a)))

tc_colors = [clf[c] for c in target_classifier]#target_classifier.map(clf)
#par_col = [p[c] for c in param]#param.map(p)
data_colors = [dt[c] for c in dataset]#list(dataset.map(dt))

row_colors = pd.DataFrame({'Target classifier':tc_colors,
                          'Dataset':data_colors,
                          #'parameters':par_col
                          })
g = sns.clustermap(wc_s1,  cmap='seismic',
                   row_colors=row_colors,
                   figsize=(5, 10),
                   col_cluster=False,
                   dendrogram_ratio=(.1, .1),
                   yticklabels=False,
                   cbar_pos=(0.99, .65, .03, .25),
                  )
plt.title("Worst case - Salem1                                                                             ")#, loc="upper left")

d_legend_lines = [Line2D([0], [0], color=colour, lw=7) for colour in dt.values()]
d_legend_names = [label for label in dt.keys()]

c_legend_lines = [Line2D([0], [0], color=colour, lw=7) for colour in clf.values()]
c_legend_names = [label for label in clf.keys()]

l1 = plt.legend(d_legend_lines, d_legend_names,
                title='Dataset', 
                loc='lower left', bbox_to_anchor=(0, -1.2))
l2 = plt.legend(c_legend_lines, c_legend_names,
                title='Target classifier', 
                loc='lower left', bbox_to_anchor=(0, -2.0))
gca().add_artist(l2)
gca().add_artist(l1)
plt.savefig("Clustermap_WorstCase_minus_Salem1.pdf")
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
#p = dict(zip(set(param), sns.color_palette(a)))
tc_colors = [clf[c] for c in target_classifier]#target_classifier.map(clf)
#par_col = [p[c] for c in param]#param.map(p)
data_colors = [dt[c] for c in dataset]#list(dataset.map(dt))
#print(data_colors)
#sns.set_palette("viridis")
row_colors = pd.DataFrame({'Target classifier':tc_colors,
                          'Dataset':data_colors,
                          #'parameters':par_col
                          })
g = sns.clustermap(wc_s2,  cmap='seismic',
                   row_colors=row_colors,
                  figsize=(5, 10),
                   col_cluster=False,
                   dendrogram_ratio=(.1, .1),
                   yticklabels=False,
                   cbar_pos=(0.99, .65, .03, .25),
                  )
plt.title("Worst case - Salem2                                                                             ")#, loc="upper left")


d_legend_lines = [Line2D([0], [0], color=colour, lw=7) for colour in dt.values()]
d_legend_names = [label for label in dt.keys()]

c_legend_lines = [Line2D([0], [0], color=colour, lw=7) for colour in clf.values()]
c_legend_names = [label for label in clf.keys()]

l1 = plt.legend(d_legend_lines, d_legend_names,
                title='Dataset', 
                loc='lower left', bbox_to_anchor=(0, -1.2))
l2 = plt.legend(c_legend_lines, c_legend_names,
                title='Target classifier', 
                loc='lower left', bbox_to_anchor=(0, -2.0))
gca().add_artist(l2)
gca().add_artist(l1)
plt.savefig("Clustermap_WorstCase_minus_Salem2.pdf")