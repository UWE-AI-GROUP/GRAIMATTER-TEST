'''
experiments.py - utilities used within experiments
'''
import pandas as pd

class ResultsEntry():
    '''
    Class that experimental results are put into. Provides them back as a dataframe
    '''
    def __init__(self, full_id, model_data_param_id, param_id,
                 dataset_name, scenario_name, classifier_name,
                 shadow_classifier_name=None, shadow_dataset=None,
                 attack_classifier_name=None,  repetition=None,
                 params=None, target_metrics=None, shadow_metrics=None, mia_metrics=None,
                 ):

        if params is None:
            params = {}
        if target_metrics is None:
            target_metrics = {}
        if shadow_metrics is None:
            shadow_metrics = {}
        if mia_metrics is None:
            mia_metrics = {}

        self.metadata = {
            'dataset': dataset_name,
            'scenario': scenario_name,
            'target_classifier': classifier_name,
            'shadow_classifier_name': shadow_classifier_name,
            'shadow_dataset': shadow_dataset,
            'attack_classifier': attack_classifier_name,
            'repetition': repetition,
            'full_id': full_id,
            'model_data_param_id': model_data_param_id,
            'param_id': param_id
        }
        self.params = params
        self.target_metrics = target_metrics
        self.shadow_metrics = shadow_metrics
        self.mia_metrics = mia_metrics
    

    def to_dataframe(self):
        '''
        Convert entry into a dataframe with a single row
        '''
        return(
            pd.DataFrame.from_dict(
                {
                    **self.metadata,
                    **self.params,
                    **self.target_metrics,
                    **self.mia_metrics,
                    **self.shadow_metrics
                }, orient='index').T
            )
