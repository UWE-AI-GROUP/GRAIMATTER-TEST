'''
experiments.py - utilities used within experiments
'''
import json
import os
import sys
import warnings
import importlib
import hashlib
import logging
import pandas as pd
from tqdm.contrib.itertools import product
import sklearn.datasets as skl_datasets

from scenarios import *
from metrics import get_metrics

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
print(PROJECT_ROOT)
sys.path.append(PROJECT_ROOT)
from data_preprocessing.data_interface import get_data_sklearn, DataNotAvailable

logger = logging.getLogger(__file__)

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


def run_loop(config_file: str) -> pd.DataFrame:
    '''
    Run the experimental loop defined in the json config_file. Return
    a dataframe of results (which is also saved as a file)
    '''

    logger.info("Running experiments with config: %s", config_file)
    with open(config_file, 'r') as f:
        config = json.loads(f.read())

    datasets = config['datasets']
    classifier_strings = config['classifiers']

    classifiers = {}
    for module_name, class_name in classifier_strings:
        module = importlib.import_module(module_name)
        class_ = getattr(module, class_name)
        classifiers[class_name] = class_

    experiment_params = config['experiment_params']

    results_filename = config['results_filename']

    n_reps = config['n_reps']

    mia_classifier_module, mia_classifier_name = config['mia_classifier']
    module = importlib.import_module(mia_classifier_module)
    mia_classifier = getattr(module, mia_classifier_name)

    scenarios = config['scenarios']

    results_df = pd.DataFrame()

    if not sys.warnoptions:
        warnings.simplefilter("once")
        #MPLClassifier is giving a lot of warnings. 
        # For each repetition are the same, so it will only show the same warning once.

    for dataset in datasets[4:]:
        logger.info("Starting datasset %s", dataset)
        #load the data
        try:
            X, y = get_data_sklearn(dataset)
        except DataNotAvailable as e:
            print(e)
            continue

        for r in range(n_reps):
            logger.info("Rep %d", r)
            #split into training, shadow model and validation data
            X_target_train, X_shadow_train, X_test, y_target_train, y_shadow_train, y_test = split_target_data(X.values, y.values, r_state=r)
            
            for classifier_name, clf_class in classifiers.items():
                logger.info("Classifier: %s", classifier_name)
                all_combinations = product(*experiment_params[classifier_name].values())
                for i, combination in enumerate(all_combinations):

                    # Turn this particular combination into a dictionary
                    params = {n: v for n, v in zip(experiment_params[classifier_name].keys(), combination)}
                    target_classifier = clf_class()
                    target_classifier.set_params(**params)

                    # Train the target model
                    target_classifier.fit(X_target_train, y_target_train.ravel())#convert that array shape to (n, ) (i.e. flatten it) -- Fix warning

                    # Get target metrics
                    target_metrics = {f"target_{key}": val for key, val in get_metrics(target_classifier, X_test, y_test).items()}
                    target_train_metrics = {f"target_train_{key}": val for key, val in get_metrics(target_classifier, X_target_train, y_target_train).items()}
                    target_metrics = {**target_metrics, **target_train_metrics}
                
                    hashstr = f'{dataset} {classifier_name} {str(params)}'
                    model_data_param_id = hashlib.sha256(hashstr.encode('utf-8')).hexdigest()
                    
                    hashstr = f'{str(params)}'
                    param_id = hashlib.sha256(hashstr.encode('utf-8')).hexdigest()
                    
                    ##########################################
                    #######   Worst case scenario     ########
                    ##########################################
                    if "WorstCase" in scenarios:
                        scenario = "WorstCase"
                        mi_test_x, mi_test_y, mi_clf = worst_case_mia(
                            target_classifier,
                            X_target_train,
                            X_test,
                            mia_classifier=mia_classifier()
                        )
                        # Get MIA metrics
                        mia_metrics = {f"mia_{key}": val for key, val in get_metrics(mi_clf, mi_test_x, mi_test_y).items()}

                        #Create ID for dataset classifier parameters scenario (but not repetition/random split)
                        hashstr = f'{dataset} {classifier_name} {str(params)} {scenario}'
                        full_id = hashlib.sha256(hashstr.encode('utf-8')).hexdigest()

                        new_results = ResultsEntry(
                            full_id, model_data_param_id, param_id,
                            dataset,
                            scenario,
                            classifier_name,
                            attack_classifier_name=mia_classifier_name,
                            repetition=r,
                            params=params,
                            target_metrics=target_metrics,
                            mia_metrics=mia_metrics
                        )

                        results_df = pd.concat([results_df, new_results.to_dataframe()], ignore_index=True)


                    ##########################################
                    #######   Salem scenario 1        ########
                    ##########################################
                    if "Salem1" in scenarios:
                        scenario = "Salem1"
                        mi_test_x, mi_test_y, mi_clf, shadow_model, X_shadow_test, y_shadow_test = salem(
                            target_classifier,
                            classifiers[classifier_name](**params),
                            X_target_train,
                            X_shadow_train,
                            y_shadow_train,
                            X_test,
                            mia_classifier=mia_classifier()
                        )

                        # Get Shadow and MIA metrics
                        shadow_metrics = {f"shadow_{key}": val for key, val in get_metrics(shadow_model, X_shadow_test, y_shadow_test).items()}
                        mia_metrics = {f"mia_{key}": val for key, val in get_metrics(mi_clf, mi_test_x, mi_test_y).items()}

                        #Create ID for dataset classifier parameters scenario (but not repetition/random split)
                        hashstr = f'{dataset} {classifier_name} {str(params)} {scenario}'
                        full_id = hashlib.sha256(hashstr.encode('utf-8')).hexdigest()

                        new_results = ResultsEntry(
                            full_id, model_data_param_id, param_id,
                            dataset,
                            scenario,
                            classifier_name,
                            shadow_dataset='Same distribution',
                            shadow_classifier_name=classifier_name,
                            attack_classifier_name=mia_classifier_name,
                            repetition=r,
                            params=params,
                            target_metrics=target_metrics,
                            mia_metrics=mia_metrics,
                            shadow_metrics=shadow_metrics
                        )

                        results_df = pd.concat([results_df, new_results.to_dataframe()], ignore_index=True)

                    ##########################################
                    #######   Salem scenario 2        ########
                    ##########################################
                    if "Salem2" in scenarios:
                        shadow_dataset = 'Breast cancer'
                        scenario = "Salem2"

                        X_breast_cancer, y_breast_cancer = skl_datasets.load_breast_cancer(return_X_y=True)

                        mi_test_x, mi_test_y, mi_clf, shadow_model, X_shadow_test, y_shadow_test = salem(
                            target_classifier,
                            classifiers[classifier_name](**params),
                            X_target_train,
                            X_breast_cancer,
                            y_breast_cancer,
                            X_test,
                            mia_classifier=mia_classifier()
                        )

                        # Get Shadow and MIA metrics
                        shadow_metrics = {f"shadow_{key}": val for key, val in get_metrics(shadow_model, X_shadow_test, y_shadow_test).items()}
                        mia_metrics = {f"mia_{key}": val for key, val in get_metrics(mi_clf, mi_test_x, mi_test_y).items()}

                        #Create ID for dataset classifier parameters scenario (but not repetition/random split)
                        hashstr = f'{dataset} {classifier_name} {str(params)} {scenario}'
                        full_id = hashlib.sha256(hashstr.encode('utf-8')).hexdigest()

                        new_results = ResultsEntry(
                            full_id, model_data_param_id, param_id,
                            dataset,
                            scenario,
                            classifier_name,
                            shadow_classifier_name=classifier_name,
                            shadow_dataset=shadow_dataset,
                            attack_classifier_name=mia_classifier_name,
                            repetition=r,
                            params=params,
                            target_metrics=target_metrics,
                            shadow_metrics=shadow_metrics,
                            mia_metrics=mia_metrics
                        )

                        results_df = pd.concat([results_df, new_results.to_dataframe()], ignore_index=True)
    
    # Save the results
    results_df.to_csv(results_filename, index=False)

def main():
    '''
    Invoke the loop
    '''
    config_file = sys.argv[1]
    run_loop(config_file)

if __name__ == '__main__':
    main()