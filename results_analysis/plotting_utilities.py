'''Plotting utilities'''

from typing import List, Tuple, Any
import pylab as plt
import pandas as pd

def filter_df(input_df: pd.DataFrame, filter: List[Tuple[str, Any]]) -> pd.DataFrame:
    '''Simple df filtering'''
    for column, value in filter:
        input_df = input_df[input_df[column] == value].copy()
    return input_df

def single_hyperparam_plot(
    results_df: pd.DataFrame,
    hyp_name: str,
    metric_name: str="mia_AUC",
    filter: List[Tuple[str, Any]]=None,
    log_hyp:bool=True) -> None:
    '''Average for a single hyper-parameter, stratified by dataset'''

    if filter is not None:
        results_df = filter_df(results_df, filter)

    datasets = results_df['dataset'].unique()
    print(datasets)
    hyp_vals = results_df[hyp_name].unique()
    print(hyp_vals)
    plt.figure(figsize=(20, 14))
    for dataset in datasets:
        plot_vals = []
        for hyp_val in hyp_vals:
            temp_df = filter_df(results_df, [("dataset", dataset), (hyp_name, hyp_val)])
            metric = temp_df[metric_name].mean()
            plot_vals.append((hyp_val, metric))
        plot_vals.sort(key = lambda x: x[0])
        x_vals, y_vals = zip(*plot_vals)
        plt.plot(x_vals, y_vals, 'o-', label=dataset)
        plt.legend()
        if log_hyp:
            plt.xscale('log')
    plt.xlabel(hyp_name)
    plt.ylabel(metric_name)
    plt.show()

def strat_kde(results_df: pd.DataFrame, hyp_name: str, metric_name: str="mia_AUC", filter_list: List[Tuple[str, Any]]=None) -> None:
    '''
    Kernel density estimate, stratified by a hyp value
    '''
    from sklearn.neighbors import KernelDensity
    import numpy as np
    results_df = filter_df(results_df, filter_list)
    hyp_vals = results_df[hyp_name].unique()
    plt.figure(figsize=(20, 14))
    x_plot = np.linspace(0.3, 1.0, 1000).reshape(-1, 1)
    alp_val = 0.1

   
    cols = [
        [0.2, 0.5, 0.7, alp_val],
        [1, 0.5, 0, alp_val],
        [0.25, 0.7, 0.25, alp_val]
    ]
    
    for i, hyp_val in enumerate(hyp_vals):
        vals_for_kde = filter_df(results_df, [(hyp_name, hyp_val)])[metric_name].values
        kde = KernelDensity(kernel='gaussian', bandwidth=0.03).fit(vals_for_kde.reshape(-1, 1))
        log_dens = kde.score_samples(x_plot)
        plt.fill(x_plot[:, 0], np.exp(log_dens), fc=cols[i])
        plt.plot(x_plot[:, 0], np.exp(log_dens), color=cols[i][:-1], label=f"{hyp_name} = {hyp_val}")
        jitter = np.random.rand(len(vals_for_kde))*0.25
        plt.plot(vals_for_kde, -jitter, '+', color=cols[i][:-1]+[0.1])
        # print(vals_for_kde)
    plt.legend()
    plt.xlabel("Membership Inference AUC")
    plt.ylabel("P(Membership Inference AUC)")
    plt.show()



if __name__ == '__main__':

    font = {'size': 22}
    plt.rc('font', **font)

    CSV_FILE = "Random_Forest_loop_results.csv"
    results_df = pd.read_csv(CSV_FILE)
    filter_list = [("scenario", "WorstCase"), ("dataset", "mimic2-iaccd")]
    # results_df.replace(
    #     {
    #         "minmax mimic2-iaccd": "mimic2-iaccd",
    #         "minmax in-hospital-mortality": "in-hospital-mortality",
    #         "minmax indian liver": "indian liver"
    #     }, inplace=True)

    strat_kde(results_df, "min_samples_split", filter_list=filter_list)
