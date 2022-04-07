"""Privacy protected Random Forest classifier."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
from dictdiffer import diff
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from ..safemodel import SafeModel


class SafeRandomForestClassifier(SafeModel, RandomForestClassifier):
    """Privacy protected Random Forest classifier."""

    def __init__(self, **kwargs: Any) -> None:
        """Creates model and applies constraints to params"""
        SafeModel.__init__(self)
        RandomForestClassifier.__init__(self, **kwargs)
        self.model_type: str = "RandomForestClassifier"
        super().preliminary_check(apply_constraints=True, verbose=True)
        self.ignore_items = [
            "model_save_file",
            "ignore_items",
            "estimators_",
            "base_estimator_",
        ]
        self.examine_seperately_items = ["base_estimator"]

    def additional_checks(
        self, curr_separate: dict, saved_separate: dict
    ) -> tuple[str, str]:
        """Random Forest-specific checks"""
        # call the super function to deal with any items that are lists
        # just in case we add any in the future
        msg, disclosive = super().additional_checks(curr_separate, saved_separate)
        # now the relevant random-forest specific things
        for item in self.examine_seperately_items:
            if item == "base_estimator":
                try:
                    the_type = type(self.base_estimator)
                    if not isinstance(self.saved_model["base_estimator_"], the_type):
                        msg += "Warning: model was fitted with different base estimator type"
                        disclosive = True
                except AttributeError:
                    msg += "Error: model has not been fitted to data"
                    disclosive = True
            elif isinstance(curr_separate[item], DecisionTreeClassifier):
                diffs_list = list(diff(curr_separate[item], saved_separate[item]))
                if len(diffs_list) > 0:
                    disclosive = True
                    msg += f"structure {item} has {len(diffs_list)} differences: {diffs_list}"
        return msg, disclosive

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """Do fit and then store model dict"""
        super().fit(x, y)
        self.saved_model = copy.deepcopy(self.__dict__)
