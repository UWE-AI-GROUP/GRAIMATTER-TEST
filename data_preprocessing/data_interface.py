'''
data_interface.py
A set of useful handlers to pull in datasets common to the project and perform the appropriate
pre-processing
'''
import os
import logging
from typing import Tuple
import pandas as pd

logging.basicConfig(
    level = "DEBUG"
)

logger = logging.getLogger(__file__)


PROJECT_ROOT_FOLDER = os.path.dirname(os.path.dirname(__file__))
logger.info("ROOT PROJECT FOLDER = %s", PROJECT_ROOT_FOLDER)


class UnknownDataset(Exception):
    '''
    If the user passes a name that we don't recognise
    '''

class DataNotAvailable(Exception):
    '''
    If the user asks for a dataset that they do not have the data for
    '''

def get_data_sklearn(
    dataset_name: str,
    data_folder: str = os.path.join(PROJECT_ROOT_FOLDER, "data")
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    '''
    Main entry method for data in format sensible for sklearn. User passes a name and that dataset
    is returned as a pandas DataFrame.

    @param dataset_name (str): the name of the dataset
    @param data_folder (str; optional): the root folder to look for data. Defaults to GRAIMatter/data
    @returns Tuple[pd.DataFrame, pd.DataFrame]: tuple of dataframes correspnding to X and y
    '''
    logger.info("DATASET FOLDER = %s", data_folder)

    if dataset_name == 'mimic2-iaccd':
        return mimic_iaccd(data_folder)
    else:
        raise UnknownDataset()


def mimic_iaccd(data_folder: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    '''
    Loads the mimic_iaccd data. We will end up with one method like this for each dataset
    (or perhaps >1 if we want to process datasets in multiple different ways)
    '''

    # Check the data has been downloaded. If not throw an exception with instructions on how to
    # download, and where to store
    file_path = os.path.join(
        data_folder,
        "mimic2-iaccd",
        "1.0",
        "full_cohort_data.csv"
    )
    if not os.path.exists(file_path):
        help_message = f"""
The MIMIC2-iaccd data is not available in {data_folder}.  
The following file should exist: {file_path}.
Please download from https://physionet.org/content/mimic2-iaccd/1.0/full_cohort_data.csv
        """
        raise DataNotAvailable(help_message)
    else:
        # File exists, load and preprocess#
        logger.info("Loading mimic2-iaccd")
        X = pd.read_csv(file_path)

        logger.info("Preprocessing")
        # remove columns non-numerical and repetitive or uninformative data for the analysis
        col = ['service_unit', 'day_icu_intime', 'hosp_exp_flg','icu_exp_flg', 'day_28_flg'] 
        # service_num is the numerical version of service_unit
        # day_icu_intime_num is the numerical version of day_icu_intime
        # the other columns are to do with death and are somewhat repetitive with censor_flg
        X.drop(col, axis = 1, inplace=True)
        # drop columns with only 1 value
        X.drop('sepsis_flg', axis=1, inplace=True)
        # drop NA by row
        X.dropna(axis=0, inplace=True)

        # extract target
        target = 'censor_flg'
        y = X[target]
        X.drop([target], axis=1, inplace=True)
        
        return X, y



if __name__ == '__main__':
    '''
    Example, if called as a script
    '''

    X, y = get_data_sklearn("mimic2-iaccd")
    print(X.head())
    print(y.head())