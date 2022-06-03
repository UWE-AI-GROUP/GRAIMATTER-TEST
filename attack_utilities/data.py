#!/usr/bin/python3

"""
Dataset loader classes.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from data_preprocessing.data_interface import get_data_sklearn

logging.basicConfig()
logger = logging.getLogger("data")
logger.setLevel(logging.INFO)


class Data:
    """Base class for storing dataset details."""

    def __init__(self) -> None:
        """Initialises empty attributes."""
        self.name: str = ""  # name of the dataset
        self.n_samples: int = 0  # number of samples in the dataset
        self.n_features: int = 0  # number of features in the dataset
        self.features: dict = {}  # dictionary description of features
        self.data: Any  # raw feature data
        self.target: Any  # raw target data
        self.x: np.ndarray  # original (unsplit) inputs
        self.y: np.ndarray  # original (unsplit) outputs
        self.Xt_member: np.ndarray  # original training set inputs
        self.yt_member: np.ndarray  # original training set outputs
        self.Xt_nonmember: np.ndarray  # original testing set inputs
        self.yt_nonmember: np.ndarray  # original testing set outputs
        self.x_train: np.ndarray  # encoded training set inputs
        self.y_train: np.ndarray  # encoded training set outputs
        self.x_test: np.ndarray  # encoded testing set inputs
        self.y_test: np.ndarray  # encoded testing set outputs

    def add_feature(self, name: str, indices: list[int], encoding: str) -> None:
        """Adds a feature description to the data dictionary."""
        index: int = len(self.features)
        self.features[index] = {
            "name": name,
            "indices": indices,
            "encoding": encoding,
        }


class NurseryData(Data):
    """Nursery dataset loading and preprocessing."""

    def __init__(self, seed: int | None) -> None:
        """Fetches the dataset and preprocesses."""
        Data.__init__(self)
        data = fetch_openml(data_id=26, as_frame=True)
        self.name = "Nursery"
        self.data = data.data
        self.target = data.target
        self.x = np.asarray(self.data, dtype=str)
        self.y = np.asarray(self.target, dtype=str)
        self.n_samples = np.shape(self.x)[0]
        self.n_features = np.shape(self.x)[1]
        indices: list[list[int]] = [
            [0, 1, 2],  # parents
            [3, 4, 5, 6, 7],  # has_nurs
            [8, 9, 10, 11],  # form
            [12, 13, 14, 15],  # children
            [16, 17, 18],  # housing
            [19, 20],  # finance
            [21, 22, 23],  # social
            [24, 25, 26],  # health
        ]
        for i in range(self.n_features):
            self.add_feature(data.feature_names[i], indices[i], "onehot")

        # target model train / test split - these are strings
        (
            self.Xt_member,
            self.Xt_nonmember,
            self.yt_member,
            self.yt_nonmember,
        ) = train_test_split(
            self.x,
            self.y,
            test_size=0.5,
            stratify=self.y,
            shuffle=True,
            random_state=seed,
        )
        # one-hot encoding of features and integer encoding of labels
        self.label_enc = LabelEncoder()
        self.feature_enc = OneHotEncoder()
        self.x_train = self.feature_enc.fit_transform(self.Xt_member).toarray()
        self.y_train = self.label_enc.fit_transform(self.yt_member)
        self.x_test = self.feature_enc.transform(self.Xt_nonmember).toarray()
        self.y_test = self.label_enc.transform(self.yt_nonmember)

        logger.info("%s", self.features)
        logger.info("x_train shape = %s", np.shape(self.x_train))
        logger.info("y_train shape = %s", np.shape(self.y_train))
        logger.info("x_test shape = %s", np.shape(self.x_test))
        logger.info("y_test shape = %s", np.shape(self.y_test))


class HospitalData(Data):
    """Nursery dataset loading and preprocessing."""

    def __init__(self, seed: int | None) -> None:
        """Fetches the dataset and preprocesses."""
        Data.__init__(self)
        self.name = "in-hospital-mortality"
        self.data, self.target = get_data_sklearn(self.name)
        #  self.data = self.data.round(3)
        self.x = np.asarray(self.data.values, dtype=np.float64)
        self.y = np.asarray(self.target.values, dtype=np.float64).ravel()
        self.n_samples = np.shape(self.x)[0]
        self.n_features = np.shape(self.x)[1]

        indices: list[list[int]] = []
        for i in range(self.n_features):
            indices.append([i])

        for i in range(self.n_features):
            self.add_feature(self.data.columns[i], indices[i], str(self.data.dtypes[i]))

        # target model train / test split - these are floats
        (
            self.Xt_member,
            self.Xt_nonmember,
            self.yt_member,
            self.yt_nonmember,
        ) = train_test_split(
            self.x,
            self.y,
            test_size=0.5,
            stratify=self.y,
            shuffle=True,
            random_state=seed,
        )
        self.x_train = self.Xt_member
        self.y_train = self.yt_member
        self.x_test = self.Xt_nonmember
        self.y_test = self.yt_nonmember

        logger.info("%s", self.features)
        logger.info("x_train shape = %s", np.shape(self.x_train))
        logger.info("y_train shape = %s", np.shape(self.y_train))
        logger.info("x_test shape = %s", np.shape(self.x_test))
        logger.info("y_test shape = %s", np.shape(self.y_test))


def get_aia_data(name: str, random_state: int | None) -> Data:
    """Returns a dataset specified by name for attribute inference."""
    if name == "nursery":
        return NurseryData(seed=random_state)
    if name == "in-hospital-mortality":
        return HospitalData(seed=random_state)
    logger.info("%s dataset not yet implemented", name)
    sys.exit()
