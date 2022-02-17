'''
data_interface.py
A set of useful handlers to pull in datasets common to the project and perform the appropriate
pre-processing
'''
import os
import logging
from typing import Tuple
import pandas as pd
import numpy as np
import pylab as plt

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
    elif dataset_name == 'in-hospital-mortality':
        return in_hospital_mortality(data_folder)
    elif dataset_name == 'medical-mnist-ab-v-br-100':
        return medical_mnist_ab_v_br_100(data_folder)
    elif dataset_name == 'indian liver':
        return indian_liver(data_folder)
    else:
        raise UnknownDataset()



def images_to_ndarray(images_dir: str, number_to_load: int, label: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    '''
    Grab number_to_load images from the images_dir and create a np array and label array
    '''
    folder_path = images_dir + os.sep
    images_names = sorted(os.listdir(folder_path))
    images_names = images_names[:number_to_load]
    np_images = np.array([plt.imread(folder_path + img).flatten() for img in images_names])
    labels = np.ones((len(np_images), 1), int) * label
    return np_images, labels

def medical_mnist_ab_v_br_100(data_folder: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    '''
    Load Medical MNIST into pandas format
    borrows heavily from: https://www.kaggle.com/harelshattenstein/medical-mnist-knn
    Creates a binary classification 
    '''

    labels_dict = {0 : 'AbdomenCT', 1 : 'BreastMRI', 2 : 'CXR', 3 : 'ChestCT', 4 : 'Hand', 5 : 'HeadCT'}

    base_folder = os.path.join(
        data_folder,
        'kaggle-medical-mnist',
        'archive',
    )

    x_ab, y_ab = images_to_ndarray(
        os.path.join(base_folder, labels_dict.get(0)),
        100,
        0
    )

    x_br, y_br = images_to_ndarray(
        os.path.join(base_folder, labels_dict.get(0)),
        100,
        1
    )

    all_x = np.vstack((x_ab, x_br))
    all_y = np.vstack((y_ab, y_br))

    return (pd.DataFrame(all_x), pd.DataFrame(all_y))


def indian_liver(data_folder: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    '''
    Indian Liver Patient Dataset
    (https://archive.ics.uci.edu/ml/datasets/ILPD+(Indian+Liver+Patient+Dataset)
    '''

    file_path = os.path.join(
        data_folder,
        "Indian Liver Patient Dataset (ILPD).csv"
    )

    if not os.path.exists(file_path):
        help_message = f"""
Data file {file_path} does not exist. Please download fhe file from:
https://archive.ics.uci.edu/ml/datasets/ILPD+(Indian+Liver+Patient+Dataset
and place it in the correct folder.
        """
        raise DataNotAvailable(help_message)

    column_names= [
        'age', 'gender', 'total Bilirubin', 'direct Bilirubin', 'Alkphos',
        'SGPT', 'SGOT', 'total proteins', 'albumin', 'A/G ratio',  'class'
    ]

    X = pd.read_csv(file_path, names=column_names, index_col=False)
    
    X.gender.replace('Male', 0, inplace=True)
    X.gender.replace('Female', 1, inplace=True)

    X.dropna(axis=0, inplace=True)

    y = X['class']
    X.drop(['class'], axis=1, inplace=True)

    return (X, y)

def in_hospital_mortality(data_folder: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    '''
    In-hospital mortality data from this study: https://datadryad.org/stash/dataset/doi:10.5061/dryad.0p2ngf1zd
    '''
    # Check the data has been downloaded. If not throw an exception with instructions on how to
    # download, and where to store
    file_path = os.path.join(
        data_folder,
        "data01.csv",
    )

    if not os.path.exists(file_path):
        help_message = f"""
Data file {file_path} does not exist. Please download the file from:
https://datadryad.org/stash/dataset/doi:10.5061/dryad.0p2ngf1zd
and place it in the correct folder.
        """
        raise DataNotAvailable(help_message)

    input_data = pd.read_csv(file_path)
    clean_data = input_data.dropna(axis=0, how='any').drop(columns=["group", "ID"])
    target = 'outcome'
    y = clean_data[target]
    X = clean_data.drop([target], axis=1)

    return (X, y)




def mimic_iaccd(data_folder: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    '''
    Loads the mimic_iaccd data and performs Alba's pre-processing
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

    # File exists, load and preprocess#
    logger.info("Loading mimic2-iaccd")
    input_data = pd.read_csv(file_path)

    logger.info("Preprocessing")
    # remove columns non-numerical and repetitive or uninformative data for the analysis
    col = ['service_unit', 'day_icu_intime', 'hosp_exp_flg','icu_exp_flg', 'day_28_flg'] 
    # service_num is the numerical version of service_unit
    # day_icu_intime_num is the numerical version of day_icu_intime
    # the other columns are to do with death and are somewhat repetitive with censor_flg
    input_data.drop(col, axis = 1, inplace=True)
    # drop columns with only 1 value
    input_data.drop('sepsis_flg', axis=1, inplace=True)
    # drop NA by row
    input_data.dropna(axis=0, inplace=True)

    # extract target
    target = 'censor_flg'
    y = input_data[target]
    X = input_data.drop([target], axis=1)
    
    return (X, y)

