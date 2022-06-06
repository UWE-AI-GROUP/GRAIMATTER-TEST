"""Privacy protected Random Forest classifier."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
from dictdiffer import diff
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from ..safemodel import SafeModel
from .safedecisiontreeclassifier import decision_trees_are_equal

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
            "base_estimator_",
        ]
        self.examine_seperately_items = ["base_estimator", "estimators_"]

    def additional_checks(
        self, curr_separate: dict, saved_separate: dict
    ) -> tuple[str, str]:
        """Random Forest-specific checks"""
    
        # call the super function to deal with any items that are lists
        msg, disclosive = super().additional_checks(curr_separate, saved_separate)
        print(f'back from superclass msg={msg}')
        # now the relevant random-forest specific things
        for item in self.examine_seperately_items:
            if item == "base_estimator":
                try:
                    the_type = type(self.base_estimator)
                    if not isinstance(self.saved_model["base_estimator_"], the_type):
                        msg += "Warning: model was fitted with different base estimator type.\n"
                        disclosive = True
                except AttributeError:
                    msg += "Error: model has not been fitted to data.\n"
                    disclosive = True
                    
            elif item == "estimators_" :   
                
                if curr_separate[item]=="Absent" and saved_separate[item]== "Absent":
                    disclosive=True
                    msg+= "Error: model has not been fitted to data.\n"
                        
                elif curr_separate[item]=="Absent":
                    disclosive=True
                    msg+= "Error: current version of model has had trees removed after fitting.\n"
                        
                elif saved_separate[item]=="Absent":
                    disclosive=True
                    msg+= "Error: current version of model has had trees manually edited.\n"
                    
                else:
                    try:    
                        num1 = len(curr_separate[item])  
                        num2 = len(saved_separate[item])
                        if(num1 != num2):
                            msg += (f"Fitted model has {num2} estimators "
                                     f"but requested version has {num1}.\n"
                                    )
                            disclosive=False
                        else:
                            for idx in range (num1):
                                same,msg2,  = decision_trees_are_equal (
                                                 curr_separate[item][idx], 
                                                 saved_separate[item][idx]
                                                 )
                                if not same:
                                     disclosive= True
                                     msg +=  f'Forest base estimators {idx} differ.'
                                     msg +=     msg2
                                    
                             
                    except BaseException as error:
                        msg += ('In SafeRandomForest.additional_checks: '
                                f'Unable to check {item} as an exception occurred: {error}.\n'
                               )
                        same=False

                
                
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
