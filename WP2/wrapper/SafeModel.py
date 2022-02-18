"""This module contains prototypes of privacy safe model wrappers."""

from __future__ import annotations

from typing import Any

import numpy as np


class SafeModel:
    """Privacy protected model base class."""

    import getpass
    import pickle

    def __init__(self) -> None:
        """Super class constructor, gets researcher name."""
        self.model_type: str = "None"
        self.model = None  
        self.model_save_file: str = "None"
        self.filename: str = "None"
        self.researcher: str = "None"
        try:
            self.researcher = self.getpass.getuser()
        except BaseException:
            self.researcher = "unknown"

    def save_model(self, name: str = "undefined") -> None:
        """Writes model to file in appropriate format."""
        self.model_save_file = name
        while self.model_save_file == "undefined":
            self.model_save_file = input(
                "Please input a name with extension for the model to be saved."
            )
        # TODO implement more file types
        if self.model_save_file[-4:] == ".pkl":
            with open(self.model_save_file, "wb") as file:
                self.pickle.dump(self.model, file)
        else:
            print("only .pkl file saves currently implemented")

    def get_constraints(self) -> dict:
        """Gets constraints relevant to the model type from the master read-only file."""
        params: dict = {}
        # TODO change to json format from text?
        with open("params.txt", "r", encoding="utf-8") as file:
            for line in file:
                contents = line.split()
                if contents[0] == self.model_type:
                    key = contents[1]
                    value = [contents[2], contents[3]]
                    params[key] = value
        return params

    def apply_constraints_old(self, **kwargs: Any) -> None:
        """Sets model attributes according to constraints. Single inheritance version"""
        params: dict = self.get_constraints()
        for key, val in kwargs.items():
            setattr(self.model, key, val)
        for key, (operator, recommended_val) in params.items():
            # TODO distinguish between ints and floats as some models take both
            # and behave differently
            if operator in ("min", "max"):
                setattr(self.model, key, int(recommended_val))
            else:
                setattr(self.model, key, recommended_val)
                
    def apply_constraints(self, **kwargs: Any) -> None:
        """Sets model attributes according to constraints. Multiple inheritance version"""
        params: dict = self.get_constraints()
        for key, val in kwargs.items():
            setattr(self, key, val)
        for key, (operator, recommended_val) in params.items():
            # TODO distinguish between ints and floats as some models take both
            # and behave differently ALSO need to  deal with not overriding safer values
            # anad not sure there is any point using setattr rather than direct assignment
            if operator in ("min", "max"):
                setattr(self, key, int(recommended_val))
            else:
                setattr(self, key, recommended_val)

    def check_model_params(self) -> tuple[str, bool]:
        """Checks whether current model parameters have been changed from
        constrained settings.Multiple inheritance version"""
        possibly_disclosive: bool = False
        msg: str = ""
        params: dict = self.get_constraints()
        for key, (operator, recommended_val) in params.items():
            current_val = str(eval(f"self.{key}"))
            # print(
            #    f"checking key {key}: current_val {current_val}, recommended {recommended_val}"
            # )
            if current_val == recommended_val:
                msg = (
                    msg
                    + f"- parameter {key} unchanged at recommended value {recommended_val}"
                )
            elif operator == "min":
                if float(current_val) > float(recommended_val):
                    msg = msg + (
                        f"- parameter {key} increased"
                        f" from recommended min value of {recommended_val} to {current_val}."
                        " This is not problematic.\n"
                    )
                else:
                    possibly_disclosive = True
                    msg = msg + (
                        f"- parameter {key} decreased"
                        f" from recommended min value of {recommended_val} to {current_val}."
                        " THIS IS POTENTIALLY PROBLEMATIC.\n"
                    )
            elif operator == "max":
                if float(current_val) < float(recommended_val):
                    msg = msg + (
                        f"- parameter {key} decreased"
                        f" from recommended max value of {recommended_val} to {current_val}."
                        " This is not problematic.\n"
                    )
                else:
                    possibly_disclosive = True
                    msg = msg + (
                        f"- parameter {key} increased"
                        f" from recommended max value of {recommended_val} to {current_val}."
                        " THIS IS POTENTIALLY PROBLEMATIC.\n"
                    )
            elif operator == "equals":
                msg = msg + (
                    f"- parameter {key} changed"
                    f" from recommended fixed value of {recommended_val} to {current_val}."
                    " THIS IS POTENTIALLY PROBLEMATIC.\n"
                )
                possibly_disclosive = True
            else:
                msg = f"- unknown operator in parameter specification {operator}"
            msg = msg + "\n"
        return msg, possibly_disclosive
    
    
    def check_model_params_old(self) -> tuple[str, bool]:
        """Checks whether current model parameters have been changed from
        constrained settings. Single inheritance version"""
        possibly_disclosive: bool = False
        msg: str = ""
        params: dict = self.get_constraints()
        for key, (operator, recommended_val) in params.items():
            current_val = str(eval(f"self.model.{key}"))
            # print(
            #    f"checking key {key}: current_val {current_val}, recommended {recommended_val}"
            # )
            if current_val == recommended_val:
                msg = (
                    msg
                    + f"- parameter {key} unchanged at recommended value {recommended_val}"
                )
            elif operator == "min":
                if float(current_val) > float(recommended_val):
                    msg = msg + (
                        f"- parameter {key} increased"
                        f" from recommended min value of {recommended_val} to {current_val}."
                        " This is not problematic.\n"
                    )
                else:
                    possibly_disclosive = True
                    msg = msg + (
                        f"- parameter {key} decreased"
                        f" from recommended min value of {recommended_val} to {current_val}."
                        " THIS IS POTENTIALLY PROBLEMATIC.\n"
                    )
            elif operator == "max":
                if float(current_val) < float(recommended_val):
                    msg = msg + (
                        f"- parameter {key} decreased"
                        f" from recommended max value of {recommended_val} to {current_val}."
                        " This is not problematic.\n"
                    )
                else:
                    possibly_disclosive = True
                    msg = msg + (
                        f"- parameter {key} increased"
                        f" from recommended max value of {recommended_val} to {current_val}."
                        " THIS IS POTENTIALLY PROBLEMATIC.\n"
                    )
            elif operator == "equals":
                msg = msg + (
                    f"- parameter {key} changed"
                    f" from recommended fixed value of {recommended_val} to {current_val}."
                    " THIS IS POTENTIALLY PROBLEMATIC.\n"
                )
                possibly_disclosive = True
            else:
                msg = f"- unknown operator in parameter specification {operator}"
            msg = msg + "\n"
        return msg, possibly_disclosive    
    
    
    

    def request_release(self, filename: str = "undefined") -> None:
        """Saves model to filename specified and creates a report for the TRE
        output checkers."""
        if filename == "undefined":
            print("You must provide the name of the file you want to save your model")
            print("For security reasons, this will overwrite previous versions")
        else:
            # resave the model
            # ideally we would then prevent over-writing
            self.filename = filename
            self.save_model(filename)
            msg, possibly_disclosive = self.check_model_params()
            outputfilename: str = self.researcher + "_checkfile.txt"
            with open(outputfilename, "a", encoding="utf-8") as file:
                file.write(
                    f"{self.researcher} created model of type {self.model_type}"
                    f" saved as {self.model_save_file}\n"
                )
                if possibly_disclosive:
                    file.write(
                        "WARNING: model has been changed"
                        f" in way that increases disclosure risk:\n{msg}\n"
                    )
                    file.write(
                        f"RECOMMENDATION: Do not allow release of file {filename}\n\n"
                    )
                else:
                    file.write(
                        f"Model has not been changed to increase risk of disclosure:\n{msg}\n"
                    )
                    file.write(
                        f"RECOMMENDATION: Run file {filename} through next step of checking procedure\n\n"
                    )

    def preliminary_check(self) -> None:
        """Allows user to test whether model parameters break safety
        constraints prior to requesting release."""
        msg, possibly_disclosive = self.check_model_params()
        if possibly_disclosive:
            print(
                "WARNING: model has been changed in way that increases disclosure risk:\n"
            )
        else:
            print(
                "Model has not been changed to increase risk of disclosure.\n"
                " These are the params:"
            )
        print(msg + "\n")

    def __str__old(self) -> str:
        """Returns string with model description. Single inheritance version"""
        return self.model_type + " with parameters: " + str(self.model.__dict__)
    
    
    def __str__(self) -> str:
        """Returns string with model description. Multiple inheritance version"""
        return self.model_type + " with parameters: " + str(self.__dict__)

    
from sklearn.tree import DecisionTreeClassifier as DecisionTreeClassifier
class SafeDecisionTree(SafeModel, DecisionTreeClassifier):    
    def __init__(self, **kwargs: Any) -> None:
        """Creates model and applies constraints to params. Multiple Inheritance Version"""
        # separate "safe_init" function
        SafeModel.__init__(self)
        DecisionTreeClassifier.__init__(self,**kwargs)
        self.model_type: str = "DecisionTreeClassifier"
        print("Safe model created using multiple inheritance")
        super().apply_constraints(**kwargs)    
    

class SafeDecisionTree_old(SafeModel):
    """Privacy protected decision tree classifier."""

    from sklearn.tree import DecisionTreeClassifier as DecisionTree

    def __init__(self, **kwargs: Any) -> None:
        """Creates model and applies constraints to params"""
        # separate "safe_init" function
        super().__init__()
        self.model_type: str = "DecisionTreeClassifier"
        self.model = self.DecisionTree()
        super().apply_constraints(**kwargs)

    def apply(self, X: np.ndarray, check_input: bool = True):  # noqa N803
        """Return the index of the leaf that each sample is predicted as."""
        return self.model.apply(X, check_input=check_input)

    def cost_complexity_pruning_path(
        self, X: np.ndarray, y: np.ndarray, sample_weight=None  # noqa N803
    ):
        """Compute the pruning path during Minimal Cost-Complexity Pruning."""
        return self.model.cost_complexity_pruning_path(
            X, y, sample_weight=sample_weight
        )

    def decision_path(self, X: np.ndarray, check_input: bool = True):  # noqa N803
        """Return the decision path in the tree."""
        return self.model.decision_path(X, check_input=check_input)

    def fit(
        self,
        X: np.ndarray,  # noqa N803
        y: np.ndarray,
        sample_weight=None,
        check_input: bool = True,
        X_idx_sorted="deprecated",  # noqa N803
    ):
        """Build a decision tree classifier from the training set (X, y)."""
        if X_idx_sorted != "deprecated":
            print("user setting of deprecated parameter X_idx_sorted ignored")
        self.model.fit(X, y, sample_weight=sample_weight, check_input=check_input)
        return self.model

    def get_depth(self) -> int:
        """Return the depth of the decision tree."""
        return self.model.get_depth()

    def get_n_leaves(self) -> int:
        """Return the number of leaves of the decision tree."""
        return self.model.get_n_leaves()

    def get_params(self, deep: bool = True) -> str:
        """Get parameters for this estimator.- AN EXAMPLE OF A METHOD BEING BLOCKED."""
        return (
            "This function is deprecated in the SafeMode class."
            " Please use the method get_params()"
        )

    def predict(self, X: np.ndarray, check_input: bool = True):  # noqa N803
        """Predict class or regression value for X."""
        return self.model.predict(X, check_input=check_input)

    def predict_log_proba(self, X: np.ndarray):  # noqa N803
        """Predict class log-probabilities of the input samples X."""
        return self.model.predict_log_proba(X)

    def predict_proba(self, X: np.ndarray, check_input: bool = True):  # noqa N803
        """Predict class probabilities of the input samples X."""
        return self.model.predict_proba(X, check_input=check_input)

    def score(self, X: np.ndarray, y: np.ndarray, sample_weight=None):  # noqa N803
        """Return the mean accuracy on the given test data and labels."""
        return self.model.score(X, y, sample_weight=sample_weight)

    def set_params(self, **params: Any) -> None:
        """Set the parameters of this estimator."""
        # TODO  check against recommendations and flag warnings here
        self.model.set_params(**params)

from sklearn.ensemble import RandomForestClassifier as RandomForest

class SafeRandomForest(SafeModel,RandomForest):
    """Privacy protected Random Forest classifier.multiple inheritance version"""

    def __init__(self, **kwargs: Any) -> None:
        """Creates model and applies constraints to params"""
        # separate "safe_init" function
        SafeModel.__init__(self)
        RandomForest.__init__(self,**kwargs)
        self.model_type: str = "RandomForestClassifier"
        super().apply_constraints(**kwargs)

    #def __getattr__(self, attr):
    #    if attr in self.__dict__:
    #        return getattr(self, attr)
    #    return getattr(self, attr)


        
class SafeRandomForest_old(SafeModel):
    """Privacy protected Random Forest classifier. Single inheritance version"""

    from sklearn.ensemble import RandomForestClassifier as RandomForest

    def __init__(self, **kwargs: Any) -> None:
        """Creates model and applies constraints to params"""
        # separate "safe_init" function
        super().__init__()
        self.model_type: str = "RandomForestClassifier"
        self.model = self.RandomForest()
        super().apply_constraints(**kwargs)

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return getattr(self, attr)
        return getattr(self.model, attr)

    def apply(self, X: np.ndarray):  # noqa N803
        """Return the index of the leaf that each sample is predicted as."""
        return self.model.apply(X)

    def decision_path(self, X: np.ndarray):  # noqa N803
        """Return the decision path in the tree."""
        return self.model.decision_path(X)

    def fit(
        self,
        X: np.ndarray,  # noqa N803
        y: np.ndarray,
        sample_weight=None,
    ):
        """Build a Random Forest classifier from the training set (X, y)."""
        self.model.fit(X, y, sample_weight=sample_weight)
        return self.model

    def predict(self, X: np.ndarray):  # noqa N803
        """Predict class or regression value for X."""
        return self.model.predict(X)

    def predict_log_proba(self, X: np.ndarray):  # noqa N803
        """Predict class log-probabilities of the input samples X."""
        return self.model.predict_log_proba(X)

    def predict_proba(self, X: np.ndarray):  # noqa N803
        """Predict class probabilities of the input samples X."""
        return self.model.predict_proba(X)

    def score(self, X: np.ndarray, y: np.ndarray, sample_weight=None):  # noqa N803
        """Return the mean accuracy on the given test data and labels."""
        return self.model.score(X, y, sample_weight=sample_weight)

    def set_params(self, **params: Any) -> None:
        """Set the parameters of this estimator."""
        # TODO  check against recommendations and flag warnings here
        self.model.set_params(**params)