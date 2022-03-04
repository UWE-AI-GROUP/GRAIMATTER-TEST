"""This module contains prototypes of privacy safe model wrappers."""

from __future__ import annotations

import getpass
import json
import pickle
from typing import Any

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


def check_min(key: str, val: Any, cur_val: Any) -> tuple[str, bool]:
    """Checks minimum value constraint."""
    if cur_val < val:
        disclosive = True
        msg = (
            f"- parameter {key} = {cur_val}"
            f" identified as less than the recommended min value of {val}."
        )
    else:
        disclosive = False
        msg = ""
    return msg, disclosive


def check_max(key: str, val: Any, cur_val: Any) -> tuple[str, bool]:
    """Checks maximum value constraint."""
    if cur_val > val:
        disclosive = True
        msg = (
            f"- parameter {key} = {cur_val}"
            f" identified as greater than the recommended max value of {val}."
        )
    else:
        disclosive = False
        msg = ""
    return msg, disclosive


def check_equal(key: str, val: Any, cur_val: Any) -> tuple[str, bool]:
    """Checks equality value constraint."""
    if cur_val != val:
        disclosive = True
        msg = (
            f"- parameter {key} = {cur_val}"
            f" identified as different than the recommended fixed value of {val}."
        )
    else:
        disclosive = False
        msg = ""
    return msg, disclosive


def removeKey(d, key):
    r = dict(d)
    del r[key]
    return r

class SafeModel:
    """Privacy protected model base class."""

    savedDict: dict = {'savedDict':'Dummy1'}
    currentDict: dict = {'savedDict':'Dummy2'}

    def __init__(self) -> None:
        """Super class constructor, gets researcher name."""
        self.model_type: str = "None"
        self.model = None
        self.model_save_file: str = "None"
        self.filename: str = "None"
        self.researcher: str = "None"
        try:
            self.researcher = getpass.getuser()
        except BaseException:
            self.researcher = "unknown"

    def save(self, name: str = "undefined") -> None:
        """Writes model to file in appropriate format."""
        self.model_save_file = name
        while self.model_save_file == "undefined":
            self.model_save_file = input(
                "Please input a name with extension for the model to be saved."
            )
        if self.model_save_file[-4:] == ".pkl":  # save to pickle
            with open(self.model_save_file, "wb") as file:
                pickle.dump(self, file)
        elif self.model_save_file[-4:] == ".sav":  # save to joblib
            joblib.dump(self, self.model_save_file)
        else:
            suffix = self.model_save_file.split(".")[-1]
            print(f"{suffix} file saves currently not supported")

    def __get_constraints(self) -> dict:
        """Gets constraints relevant to the model type from the master read-only file."""
        rules: dict = {}
        with open("rules.json", "r", encoding="utf-8") as json_file:
            parsed = json.load(json_file)
            rules = parsed[self.model_type]
        return rules["rules"]

    def __check_model_param(
        self, rule: dict, apply_constraints: bool
    ) -> tuple[str, bool]:
        """Checks whether a current model parameter violates a safe rule.
        Optionally fixes violations."""
        disclosive: bool = False
        msg: str = ""
        operator: str = rule["operator"]
        key: str = rule["keyword"]
        val: Any = rule["value"]
        cur_val: Any = getattr(self, key)
        if operator == "min":
            msg, disclosive = check_min(key, val, cur_val)
        elif operator == "max":
            msg, disclosive = check_max(key, val, cur_val)
        elif operator == "equals":
            msg, disclosive = check_equal(key, val, cur_val)
        else:
            msg = f"- unknown operator in parameter specification {operator}"
        if apply_constraints and disclosive:
            setattr(self, key, val)
            msg += f"\nChanged parameter {key} = {val}.\n"
        return msg, disclosive

    def __check_model_param_and(
        self, rule: dict, apply_constraints: bool
    ) -> tuple[str, bool]:
        """Checks whether current model parameters violate a logical AND rule.
        Optionally fixes violations."""
        disclosive: bool = False
        msg: str = ""
        for arg in rule["subexpr"]:
            op_msg, op_disclosive = self.__check_model_param(arg, apply_constraints)
            msg += op_msg
            if op_disclosive:
                disclosive = True
        return msg, disclosive

    def __check_model_param_or(self, rule: dict) -> tuple[str, bool]:
        """Checks whether current model parameters violate a logical OR rule."""
        disclosive: bool = True
        msg: str = ""
        for arg in rule["subexpr"]:
            op_msg, op_disclosive = self.__check_model_param(arg, False)
            msg += op_msg
            if not op_disclosive:
                disclosive = False
        return msg, disclosive

    def preliminary_check(
        self, verbose: bool = True, apply_constraints: bool = False
    ) -> tuple[str, bool]:
        """Checks whether current model parameters violate the safe rules.
        Optionally fixes violations."""
        disclosive: bool = False
        msg: str = ""
        rules: dict = self.__get_constraints()
        for rule in rules:
            operator = rule["operator"]
            if operator == "and":
                op_msg, op_disclosive = self.__check_model_param_and(
                    rule, apply_constraints
                )
            elif operator == "or":
                op_msg, op_disclosive = self.__check_model_param_or(rule)
            else:
                op_msg, op_disclosive = self.__check_model_param(
                    rule, apply_constraints
                )
            msg += op_msg
            if op_disclosive:
                disclosive = True
        if disclosive:
            msg = "WARNING: model parameters may present a disclosure risk:\n" + msg
        else:
            msg = "Model parameters are within recommended ranges.\n" + msg
        if verbose:
            print(msg)
        return msg, disclosive

    def request_release(self, filename: str = "undefined") -> None:
        """Saves model to filename specified and creates a report for the TRE
        output checkers."""

        #print(self.currentDict)
        self.currentDict = self.__dict__
        #print(self.currentDict)
        self.currentDict = removeKey(self.currentDict,"currentDict")
        self.currentDict = removeKey(self.currentDict,"savedDict")

        #print(self.savedDict)
        self.savedDict = removeKey(self.savedDict,"currentDict")
        self.savedDict = removeKey(self.savedDict,"savedDict")
        #print(self.savedDict)
        
        if(self.currentDict != self.savedDict):
            print("Model Parameters do not match those used to fit the model.")
            print("You must fit the model before release.")
            print("currentDict: "+ self.currentDict)
            print("savedDict: " + self.savedDict)
            
        elif filename == "undefined":
            print("You must provide the name of the file you want to save your model")
            print("For security reasons, this will overwrite previous versions")
        else:
            print("currentDict and savedDict Match")
            print("currentDict: "+ str(self.currentDict))
            print("savedDict: " + str(self.savedDict))
            self.save(filename)
            msg, disclosive = self.preliminary_check(verbose=False)
            output: dict = {
                "researcher": self.researcher,
                "model_type": self.model_type,
                "model_save_file": self.model_save_file,
                "details": msg,
            }
            if disclosive:
                output["recommendation"] = "Do not allow release"
            else:
                output[
                    "recommendation"
                ] = f"Run file {filename} through next step of checking procedure"
            json_str = json.dumps(output, indent=4)
            outputfilename = self.researcher + "_checkfile.json"
            with open(outputfilename, "a", encoding="utf-8") as file:
                file.write(json_str)

    def __str__(self) -> str:
        """Returns string with model description."""
        return self.model_type + " with parameters: " + str(self.__dict__)


class SafeDecisionTree(SafeModel, DecisionTreeClassifier):
    """Privacy protected Decision Tree classifier."""

    def __init__(self, **kwargs: Any) -> None:
        """Creates model and applies constraints to params."""
        SafeModel.__init__(self)
        DecisionTreeClassifier.__init__(self, **kwargs)
        self.model_type: str = "DecisionTreeClassifier"
        super().preliminary_check(apply_constraints=True, verbose=True)

    def fit(self, X, y):
        """Do fit and then store model dict"""
        super().fit(X, y)
        self.savedDict = self.__dict__

class SafeRandomForest(SafeModel, RandomForestClassifier):
    """Privacy protected Random Forest classifier."""

    def __init__(self, **kwargs: Any) -> None:
        """Creates model and applies constraints to params"""
        SafeModel.__init__(self)
        RandomForestClassifier.__init__(self, **kwargs)
        self.model_type: str = "RandomForestClassifier"
        super().preliminary_check(apply_constraints=True, verbose=True)

    def fit(self, **kwargs: Any) -> None:
        """Do fit and then store model dict"""
        super().fit(self, **kwargs)
        self.savedDict = self.__dict__

    # def __getattr__(self, attr):
    #    if attr in self.__dict__:
    #        return getattr(self, attr)
    #    return getattr(self, attr)
