'''
data_interface.py
A set of useful handlers to pull in datasets common to the project and perform the appropriate
pre-processing
'''
import os, json
import logging
from typing import Tuple
import pandas as pd
import numpy as np
import pylab as plt
from zipfile import ZipFile
from collections import Counter

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
    elif dataset_name == 'texas hospitals 10':
        return texas_hospitals(data_folder)
    else:
        raise UnknownDataset()



def images_to_ndarray(images_dir: str, number_to_load: int, label: int) -> Tuple[np.array, np.array]:
    '''
    Grab number_to_load images from the images_dir and create a np array and label array
    '''
    folder_path = images_dir + os.sep
    images_names = sorted(os.listdir(folder_path))
    images_names = images_names[:number_to_load]
    np_images = np.array([plt.imread(folder_path + img).flatten() for img in images_names])
    labels = np.ones((len(np_images), 1), int) * label
    return(np_images, labels)

def medical_mnist_ab_v_br_100(data_folder: str) -> Tuple[np.array, np.array]:
    '''
    Load Medical MNIST into pandas format
    borrows heavily from: https://www.kaggle.com/harelshattenstein/medical-mnist-knn
    Creates a binary classification 
    '''
    
    base_folder = os.path.join(
        data_folder,
        'kaggle-medical-mnist',
        'archive',
    )
    
    zip_file = os.path.join(
        data_folder,
        'kaggle-medical-mnist',
        "archive.zip"
    )
    
    print(base_folder, data_folder)
    if not any([os.path.exists(base_folder), os.path.exists(zip_file)]):
        help_message = f"""
Data file {base_folder} does not exist. Please download fhe file from:
https://www.kaggle.com/andrewmvd/medical-mnist 
and place it in the correct folder. It unzips the file first.
        """
        raise DataNotAvailable(help_message)
    
    elif os.path.exists(base_folder):
        pass
    elif os.path.exists(zip_file):
        try:
            with ZiplFile(zip_file) as z:
                z.extractall()
                print("Extracted all")
                #os.remove(zip_file)
                #print("zip file removed")
        except:
            print("Invalid file")
    
    labels_dict = {0 : 'AbdomenCT', 1 : 'BreastMRI', 2 : 'CXR', 3 : 'ChestCT', 4 : 'Hand', 5 : 'HeadCT'}

    

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

    return (np.array(all_x), np.array(all_y))


def indian_liver(data_folder: str) -> Tuple[np.array, np.array]:
    '''
    Indian Liver Patient Dataset
     https://archive.ics.uci.edu/ml/machine-learning-databases/00225/Indian%20Liver%20Patient%20Dataset%20(ILPD).csv
    '''
    #(https://archive.ics.uci.edu/ml/datasets/ILPD+(Indian+Liver+Patient+Dataset)
    file_path = os.path.join(
        data_folder,
        "Indian Liver Patient Dataset (ILPD).csv"
    )
    print(file_path, data_folder)
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

    return (np.array(X), np.array(y))

def in_hospital_mortality(data_folder: str) -> Tuple[np.array, np.array]:
    '''
    In-hospital mortality data from this study: https://datadryad.org/stash/dataset/doi:10.5061/dryad.0p2ngf1zd
    '''
    # Check the data has been downloaded. If not throw an exception with instructions on how to
    # download, and where to store
    files = ["data01.csv", "doi_10.5061_dryad.0p2ngf1zd__v5.zip"]
    file_path = [os.path.join(data_folder, f) for f in files]

    if not any([os.path.exists(fp) for fp in file_path]):
        help_message = f"""
Data file {file_path[0]} or {file_path[1]} does not exist. Please download the file from:
https://datadryad.org/stash/dataset/doi:10.5061/dryad.0p2ngf1zd
and place it in the correct folder. It works with either the zip file or uncompressed.
        """
        raise DataNotAvailable(help_message)
    
    if os.path.exists(file_path[1]):
        input_data = pd.read_csv(ZipFile(file_path[1]).open("data01.csv"))
    else:
        input_data = pd.read_csv(file_path)
    clean_data = input_data.dropna(axis=0, how='any').drop(columns=["group", "ID"])
    target = 'outcome'
    y = clean_data[target]
    X = clean_data.drop([target], axis=1)

    return (np.array(X), np.array(y))




def mimic_iaccd(data_folder: str) -> Tuple[np.array, np.array]:
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


def texas_hospitals(data_folder: str) -> Tuple[np.array, np.array]:
    '''
    Texas Hospitals Dataset
    (https://www.dshs.texas.gov/THCIC/Hospitals/Download.shtm)
    '''
    file_list = ["PUDF_1Q2006_tab-delimited.zip", 
                 "PUDF_1Q2007_tab-delimited.zip", 
                 "PUDF_1Q2008_tab-delimited.zip", 
                 "PUDF_1Q2009_tab-delimited.zip", 
                 "PUDF_2Q2006_tab-delimited.zip", 
                 "PUDF_2Q2007_tab-delimited.zip", 
                 "PUDF_2Q2008_tab-delimited.zip", 
                 "PUDF_2Q2009_tab-delimited.zip", 
                 "PUDF_3Q2006_tab-delimited.zip", 
                 "PUDF_3Q2007_tab-delimited.zip", 
                 "PUDF_3Q2008_tab-delimited.zip", 
                 "PUDF_3Q2009_tab-delimited.zip", 
                 "PUDF_4Q2006_tab-delimited.zip", 
                 "PUDF_4Q2007_tab-delimited.zip", 
                 "PUDF_4Q2008_tab-delimited.zip", 
                 "PUDF_4Q2009_tab-delimited.zip"]
    
    files_path = [os.path.join(
        data_folder,"TexasHospitals",f) for f in file_list]
    
    found = [os.path.exists(file_path) for file_path in files_path]
    not_found = [file_path for file_path in files_path if not os.path.exists(file_path)]
    #not_found = [file_path for m in [i for i,x in enumerate(found) if x==False]]
    if not all(found):
        help_message = f"""
    Some or all data files do not exist. Please accept their terms & conditions, then download the tab delimited files from each quarter during 2006-2009 from:
    https://www.dshs.texas.gov/THCIC/Hospitals/Download.shtm
and place it in the correct folder.

    Missing files are:
    {not_found}
        """
        raise DataNotAvailable(help_message)
    elif not os.path.exists(os.path.join(data_folder,"TexasHospitals","texas_data10.csv")):
    
        logger.info("Processing Texas Hospitals data (2006-2009)")

        #Load data
        columns_names = ['THCIC_ID',# Provider ID. Unique identifier assigned to the provider by DSHS. Hospitals with fewer than 50 discharges have been aggregated into the Provider ID '999999'
                     'DISCHARGE_QTR', #yyyyQm
                     'TYPE_OF_ADMISSION',
                     'SOURCE_OF_ADMISSION',
                     'PAT_ZIP',#Patient’s five-digit ZIP code
                     'PUBLIC_HEALTH_REGION', #Public Health Region of patient’s address
                     'PAT_STATUS', #Code indicating patient status as of the ending date of service for the period of care reported
                     'SEX_CODE',
                     'RACE',
                     'ETHNICITY',
                     'LENGTH_OF_STAY',
                     'PAT_AGE', #Code indicating age of patient in days or years on date of discharge. 
                     'PRINC_DIAG_CODE', #diagnosis code for the principal diagnosis
                     'E_CODE_1', #external cause of injury
                     'PRINC_SURG_PROC_CODE', #Code for the principal surgical or other procedure performed during the period covered by the bill           
                     'RISK_MORTALITY', #Assignment of a risk of mortality score from the All Patient Refined (APR) Diagnosis Related Group (DRG) 
                     'ILLNESS_SEVERITY',#Assignment of a severity of illness score from the All Patient Refined (APR) Diagnosis RelatedGroup (DRG
                     'RECORD_ID'
                    ]
        #obtain the 100 most frequent procedures
        tmp = []
        for f in files_path:
            df = [pd.read_csv(ZipFile(f).open(i), sep="\t", usecols=["PRINC_SURG_PROC_CODE"]) for i in ZipFile(f).namelist() if 'base' in i][0]
            df.dropna(inplace=True)
            tmp.extend(list(df.PRINC_SURG_PROC_CODE))
        princ_surg_proc_keep = [k for k,v in Counter(tmp).most_common(10)]
        #remove unecessary variables
        del tmp

        #Load the data    
        tx_data = pd.DataFrame()
        for f in files_path:
            df = [pd.read_csv(ZipFile(f).open(i), sep="\t", usecols=columns_names)
                  for i in ZipFile(f).namelist() if 'base' in i][0]
            #keep only those rows with one of the 10 most common principal surgical procedure
            df = df[df["PRINC_SURG_PROC_CODE"].isin(princ_surg_proc_keep)]
            #clean up data
            df.dropna(inplace=True)
            df.replace('`', pd.NA, inplace=True)
            df.replace('*', pd.NA, inplace=True)
            #replace sex to numeric
            df.SEX_CODE.replace('M',0,inplace=True)
            df.SEX_CODE.replace('F',1,inplace=True)
            df.SEX_CODE.replace('U',2,inplace=True)
            #set to numerical variable
            [df.DISCHARGE_QTR.replace(d_code, ''.join(d_code.split('Q')), inplace=True) for d_code in set(list(df.DISCHARGE_QTR))]
            df.dropna(inplace=True)
            #merge data
            tx_data = pd.concat([tx_data, df])
        #remove uncessary variables
        del df

        #renumber non-numerical codes for cols
        cols=['PRINC_DIAG_CODE', 'SOURCE_OF_ADMISSION', 'E_CODE_1']
        for col in cols:
            tmp = list(set([x for x in tx_data[col] if not str(x).isdigit() and not isinstance(x, float)]))
            n = max(list(set([int(x) for x in tx_data[col] if str(x).isdigit() or isinstance(x, float)])))
            [tx_data[col].replace(x, n+i,inplace=True) for i,x in enumerate(tmp)]   
        del tmp, n
        #set index
        tx_data.set_index('RECORD_ID', inplace=True)
        #final check and drop of NAs
        tx_data.dropna(inplace=True)
        #convert all data to numerical
        tx_data = tx_data.astype(int)
        #save csv file
        tx_data.to_csv(os.path.join(data_folder, "TexasHospitals", "texas_data10.csv"))
    else:
        logger.info("Loading processed Texas Hospitals data (2006-2009) csv file.")
        #load texas data processed csv file
        tx_data = pd.read_csv(os.path.join(data_folder, "TexasHospitals", "texas_data10.csv"))
        
    # extract target
    var = 'RISK_MORTALITY'
    labels = tx_data[var]
    # Drop the column that contains the labels
    tx_data.drop([var], axis=1, inplace=True)
    
    return(np.array(tx_data), np.array(labels))