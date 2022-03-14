'''
results_loader.py -- code to help with loading and analysing the results .csv files
'''

import sys
import json
import logging
from typing import Tuple
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

IMPUTE_VALS = (
    ('max_depth', 200),
)

def load_results_csv(
    config_filename: str,
    one_hot_encode: bool=True,
    impute_missing: bool=True,
    impute_vals: Tuple = IMPUTE_VALS) -> pd.DataFrame:
    '''
    load_results_csv
    Load a results csv file from the associated config. Options for imputing missing
    @param config_filename the full path to the config file
    @param one_hot_encode whether or not to encode non numerical hyper-pars with one-hot-encoding
    @param impute_missing boolean (default = True), imputes missing values. Only imputes where
    the column name is stored in the impute_vals tuple
    @param impute_vals: tuple of tuples. Each subtuple has a hyper-param name and the value to
    impute. This should be used for numerical hyper-parameters that can be set to null and which
    shouldn't be treated as categorical. E.g. max_depth in the RandomForest
    '''

    # pylint: disable = unsubscriptable-object


    logging.info("Loading config: %s", config_filename)
    # load the config
    with open(config_filename, 'r', encoding='utf-8') as file_handle:
        config = json.load(file_handle)

    logging.info("Attempting to load results from %s", config['results_filename'])
    results_df = pd.read_csv(config['results_filename'])

    logging.info("Loaded %d rows", len(results_df))

    if impute_missing:
        logging.info("Imputing missing values, will skip shadow columns")
        # Convert impute_vals to a dictionary
        impute_vals = {
            sub_tuple[0]: sub_tuple[1] for sub_tuple in impute_vals
        }
        for column in results_df.columns:
            if any(results_df[column].isna()):
                if column in impute_vals:
                    results_df[column][results_df[column].isna()] = impute_vals[column]
                    logging.info("Replaced NAs in %s with %s", column, str(impute_vals[column]))
                else:
                    logging.info("Found NA in %s, but no default in impute_vals, skipping", column)



    logging.info("Extracting hyper-par names from config")
    hyper_pars = []
    for classifier_name in config['experiment_params']:
        for hyper_par in config['experiment_params'][classifier_name]:
            hyper_pars.append(hyper_par)
    hyper_pars = set(hyper_pars)

    logging.info("Found: %s", ", ".join(hyper_pars))

    # Get the datatypes of the columns
    column_dtype = results_df.dtypes # pylint: disable = no-member

    if one_hot_encode:
        logging.info("One hot encoding relevant hyper-pars")

        # Find the non-numeric hyper-parameters
        cat_cols = []
        for _, col_name in enumerate(hyper_pars):
            d_type = column_dtype[col_name]
            if d_type in ("int64", "float64"):
                continue
            cat_cols.append(col_name)
        logging.info("Will encode %s", ", ".join(cat_cols))

        # Do the endoding
        one_hot_encoder = OneHotEncoder()
        results_df.drop(cat_cols, axis=1, inplace=True)
        encoded = pd.DataFrame(
            one_hot_encoder.fit_transform(results_df[cat_cols].copy()).toarray(),
            columns=one_hot_encoder.get_feature_names_out()
        )

        # Merge the binary features back in
        encoded['full_id'] = results_df['full_id'].copy()
        nrow_check = len(results_df)
        results_df = results_df.merge(encoded, how='left', on='full_id')
        logging.info("One hot encoding done, row counts match = %s", nrow_check == len(results_df))

    return results_df
