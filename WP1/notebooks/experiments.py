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
import argparse
import pandas as pd
from tqdm.contrib.itertools import product
import sklearn.datasets as skl_datasets


from scenarios import worst_case_mia, salem, split_target_data # pylint: disable=import-error
from metrics import get_metrics # pylint: disable=import-error

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(PROJECT_ROOT)
from data_preprocessing.data_interface import get_data_sklearn, DataNotAvailable
from WP1.notebooks.scenarios import *
from WP1.notebooks.metrics import get_metrics

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


def run_loop(config_file: str, append: bool) -> pd.DataFrame:
    '''
    Run the experimental loop defined in the json config_file. Return
    a dataframe of results (which is also saved as a file)
    '''

    logger.info("Running experiments with config: %s", config_file)
    with open(config_file, 'r', encoding='utf-8') as config_handle:
        config = json.loads(config_handle.read())

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

    if not sys.warnoptions:
        warnings.simplefilter("once")
        #MPLClassifir is giving a lot of warnings.
        # For each repetition are the same, so it will only show the same warning once.

    if append:
        #load full_id from results file to check whether certains combinations already exists.
        results_df = pd.read_csv(results_filename)
        existing_experiments = set(
            [
                (fid, rep) for fid,rep in zip(
                    results_df['model_data_param_id'], results_df['repetition']
                )
            ]
        )
    else:
        results_df = pd.DataFrame()


    for dataset in datasets:
        logger.info("Starting datasset %s", dataset)
        #load the data
        try:
            data_features, data_labels = get_data_sklearn(dataset)
        except DataNotAvailable as data_exception:
            logger.error(data_exception)
            continue

        for repetition in range(n_reps):
            logger.info("Rep %d", repetition)
            #split into training, shadow model and validation data
            x_target_train, x_shadow_train, x_test, y_target_train, y_shadow_train, y_test = \
                split_target_data(
                    data_features.values,
                    data_labels.values.flatten(),
                    r_state=repetition
                )

            for classifier_name, clf_class in classifiers.items():
                logger.info("Classifier: %s", classifier_name)
                all_combinations = product(*experiment_params[classifier_name].values())
                for _, combination in enumerate(all_combinations):
                    # Turn this particular combination into a dictionary
                    params = {n: v for n, v in \
                        zip(experiment_params[classifier_name].keys(), combination)}

                    hashstr = f'{dataset} {classifier_name} {str(params)}'
                    model_data_param_id = hashlib.sha256(hashstr.encode('utf-8')).hexdigest()

                    #check if this already exist in results file when append==True
                    if append and (model_data_param_id, repetition) not in existing_experiments:
                        hashstr = f'{str(params)}'
                        param_id = hashlib.sha256(hashstr.encode('utf-8')).hexdigest()

                        target_classifier = clf_class()
                        target_classifier.set_params(**params)

                        # Train the target model
                        target_classifier.fit(x_target_train, y_target_train)

                        # Get target metrics
                        target_metrics = {f"target_{key}": val for key, val in \
                            get_metrics(target_classifier, x_test, y_test).items()}
                        target_train_metrics = {f"target_train_{key}": val for key, val in \
                            get_metrics(target_classifier, x_target_train, y_target_train).items()}
                        target_metrics = {**target_metrics, **target_train_metrics}

                        ##########################################
                        #######   Worst case scenario     ########
                        ##########################################

                        if "WorstCase" in scenarios:
                            scenario = "WorstCase"
                            mi_test_x, mi_test_y, mi_clf = worst_case_mia(
                                target_classifier,
                                x_target_train,
                                x_test,
                                mia_classifier=mia_classifier()
                            )
                            # Get MIA metrics
                            mia_metrics = {f"mia_{key}": val for key, val in \
                                get_metrics(mi_clf, mi_test_x, mi_test_y).items()}

                            # Create ID for dataset classifier parameters scenario
                            #(but not repetition/random split)
                            hashstr = f'{dataset} {classifier_name} {str(params)} {scenario}'
                            full_id = hashlib.sha256(hashstr.encode('utf-8')).hexdigest()

                            new_results = ResultsEntry(
                                full_id, model_data_param_id, param_id,
                                dataset,
                                scenario,
                                classifier_name,
                                attack_classifier_name=mia_classifier_name,
                                repetition=repetition,
                                params=params,
                                target_metrics=target_metrics,
                                mia_metrics=mia_metrics
                            )

                            results_df = pd.concat(
                                [results_df, new_results.to_dataframe()],
                                ignore_index=True
                            )


                        ##########################################
                        #######   Salem scenario 1        ########
                        ##########################################

                        if "Salem1" in scenarios:
                            scenario = "Salem1"
                            mi_test_x, mi_test_y, mi_clf, shadow_model, x_shadow_test, y_shadow_test = salem( # pylint: disable = line-too-long
                                target_classifier,
                                classifiers[classifier_name](**params),
                                x_target_train,
                                x_shadow_train,
                                y_shadow_train,
                                x_test,
                                mia_classifier=mia_classifier()
                            )

                            # Get Shadow and MIA metrics
                            shadow_metrics = {f"shadow_{key}": val for key, val in \
                                get_metrics(shadow_model, x_shadow_test, y_shadow_test).items()}
                            mia_metrics = {f"mia_{key}": val for key, val in \
                                get_metrics(mi_clf, mi_test_x, mi_test_y).items()}

                            # Create ID for dataset classifier parameters scenario
                            # (but not repetition/random split)
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
                                repetition=repetition,
                                params=params,
                                target_metrics=target_metrics,
                                mia_metrics=mia_metrics,
                                shadow_metrics=shadow_metrics
                            )

                            results_df = pd.concat(
                                [results_df, new_results.to_dataframe()],
                                ignore_index=True
                            )

                        ##########################################
                        #######   Salem scenario 2        ########
                        ##########################################

                        if "Salem2" in scenarios:
                            shadow_dataset = 'Breast cancer'
                            scenario = "Salem2"

                            x_breast_cancer, y_breast_cancer = skl_datasets.load_breast_cancer(return_X_y=True) # pylint: disable = line-too-long

                            mi_test_x, mi_test_y, mi_clf, shadow_model, x_shadow_test, y_shadow_test = salem( # pylint: disable = line-too-long
                                target_classifier,
                                classifiers[classifier_name](**params),
                                x_target_train,
                                x_breast_cancer,
                                y_breast_cancer,
                                x_test,
                                mia_classifier=mia_classifier()
                            )

                            # Get Shadow and MIA metrics
                            shadow_metrics = {f"shadow_{key}": val for key, val in \
                                get_metrics(shadow_model, x_shadow_test, y_shadow_test).items()}
                            mia_metrics = {f"mia_{key}": val for key, val in \
                                get_metrics(mi_clf, mi_test_x, mi_test_y).items()}

                            # Create ID for dataset classifier parameters scenario
                            # (but not repetition/random split)
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
                                repetition=repetition,
                                params=params,
                                target_metrics=target_metrics,
                                shadow_metrics=shadow_metrics,
                                mia_metrics=mia_metrics
                            )

                            results_df = pd.concat(
                                [results_df, new_results.to_dataframe()],
                                ignore_index=True
                            )
            # Save after each repetition
            results_df.to_csv(results_filename, index=False)

def main():
    '''
    Invoke the loop
    '''
    parser = argparse.ArgumentParser(description=(
        'Run predictions with the parameters defined in the config file.'
        ' Default: overwrite results file.'
        )
    )
    parser.add_argument(
        action='store',
        dest='config_filename',
        help=(
            'json formatted file that contain hyper-parameter for loop search. '
            'It is assumed the file is located in "experiments" directory, so please provide path '
            'and filename, e.g. RF/randomForest_config.json'
        )
    )
    parser.add_argument(
        '--append',
        action='store_true',
        help=(
            'It checks if there is a results file and checks which combination of '
            'hyper-parameters need to run. Default: append=False.'
        )
    )

    args = parser.parse_args()
    config_file = args.config_filename
    append = args.append

    run_loop(config_file, append)

if __name__ == '__main__':
    main()
