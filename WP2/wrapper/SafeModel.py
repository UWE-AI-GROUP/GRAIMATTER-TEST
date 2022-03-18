"""This module contains prototypes of privacy safe model wrappers."""

from __future__ import annotations

import copy
import getpass
import json
import pickle
from typing import Any

import joblib
import numpy as np
from dictdiffer import diff
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree._tree import Tree


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


def check_type(key: str, val: Any, cur_val: Any) -> tuple[str, bool]:
    """Checks the type of a value."""
    if type(cur_val).__name__ != val:
        disclosive = True
        msg = (
            f"- parameter {key} = {cur_val}"
            f" identified as different type to recommendation of {val}."
        )
    else:
        disclosive = False
        msg = ""
    return msg, disclosive


class SafeModel:
    """Privacy protected model base class."""

    def __init__(self) -> None:
        """Super class constructor, gets researcher name."""
        self.model_type: str = "None"
        self.model = None
        self.saved_model = None
        self.model_save_file: str = "None"
        self.ignore_items: list[str] = []
        self.examine_seperately_items: list[str] = []
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

    def __apply_constraints(
        self, operator: str, key: str, val: Any, cur_val: Any
    ) -> str:
        """Applies a safe rule for a given parameter."""
        if operator == "is_type":
            if (val == "int") and (type(cur_val).__name__ == "float"):
                self.__dict__[key] = int(self.__dict__[key])
                msg = f"\nChanged parameter type for {key} to {val}.\n"
            elif (val == "float") and (type(cur_val).__name__ == "int"):
                self.__dict__[key] = float(self.__dict__[key])
                msg = f"\nChanged parameter type for {key} to {val}.\n"
            else:
                msg = (
                    f"Nothing currently implemented to change type of parameter {key} "
                    f"from {type(cur_val).__name__} to {val}.\n"
                )
        else:
            setattr(self, key, val)
            msg = f"\nChanged parameter {key} = {val}.\n"
        return msg

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
        elif operator == "is_type":
            msg, disclosive = check_type(key, val, cur_val)
        else:
            msg = f"- unknown operator in parameter specification {operator}"
        if apply_constraints and disclosive:
            msg += self.__apply_constraints(operator, key, val, cur_val)
        return msg, disclosive

    def __check_model_param_and(
        self, rule: dict, apply_constraints: bool
    ) -> tuple[str, bool]:
        """Checks whether current model parameters violate a logical AND rule.
        Optionally fixes violations."""
        disclosive: bool = False
        msg: str = ""
        for arg in rule["subexpr"]:
            m, d = self.__check_model_param(arg, apply_constraints)
            msg += m
            if d:
                disclosive = True
        return msg, disclosive

    def __check_model_param_or(self, rule: dict) -> tuple[str, bool]:
        """Checks whether current model parameters violate a logical OR rule."""
        disclosive: bool = True
        msg: str = ""
        for arg in rule["subexpr"]:
            m, d = self.__check_model_param(arg, False)
            msg += m
            if not d:
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
                m, d = self.__check_model_param_and(rule, apply_constraints)
            elif operator == "or":
                m, d = self.__check_model_param_or(rule)
            else:
                m, d = self.__check_model_param(rule, apply_constraints)
            msg += m
            if d:
                disclosive = True
        if disclosive:
            msg = "WARNING: model parameters may present a disclosure risk:\n" + msg
        else:
            msg = "Model parameters are within recommended ranges.\n" + msg
        if verbose:
            print(msg)
        return msg, disclosive

    def posthoc_check(self) -> tuple[str, str]:
        """Checks whether model has been interfered with since fit() was last run"""

        disclosive = False
        msg = ""
        # get dictionaries of parameters
        current_model = copy.deepcopy(self.__dict__)
        saved_model = current_model.pop("saved_model", "Absent")

        if saved_model == "Absent":
            msg = "Error: user has not called fit() method or has deleted saved values."
            msg += "Recommendation: Do not release."
            disclosive = True

        else:
            saved_model = dict(self.saved_model)
            # in case fit has been called twice
            _ = saved_model.pop("saved_model", "Absent")

            # remove things we don't care about
            for item in self.ignore_items:
                _ = current_model.pop(item, "Absent")
                _ = saved_model.pop(item, "Absent")
            # break out things that need to be examined in more depth
            # and keep in separate lists
            curr_separate = {}
            saved_separate = {}
            for item in self.examine_seperately_items:
                curr_separate[item] = current_model.pop(item, "Absent")
                saved_separate[item] = saved_model.pop(item, "Absent")

            # comparison on list of "simple" parameters
            match = list(diff(current_model, saved_model, expand=True))
            if len(match) > 0:
                disclosive = True
                msg += f"Warning: basic parameters differ in {len(match)} places:\n"
                for i in range(len(match)):
                    if match[i][0] == "change":
                        msg += f"parameter {match[i][1]} changed from {match[i][2][1]} "
                        msg += f"to {match[i][2][0]} after model was fitted\n"
                    else:
                        msg += f"{match[i]}"

            # comparison of more complex structures
            # in the super class we just check these model-specific items exist
            # in both current and saved copies
            for item in self.examine_seperately_items:
                if curr_separate[item] == "Absent" and saved_separate[item] == "Absent":
                    # not sure if this is necessarily disclosive
                    msg += f"Note that item {item} missing from both versions"

                elif (curr_separate[item] == "Absent") and not (
                    saved_separate[item] == "Absent"
                ):
                    disclosive = True
                    msg += f"Error, item {item} present in  saved but not current model"
                elif (saved_separate[item] == "Absent") and not (
                    curr_separate[item] == "Absent"
                ):
                    disclosive = True
                    msg += f"Error, item {item} present in current but not saved model"
                else:
                    msg2, disclosive2 = self.additional_checks(
                        curr_separate, saved_separate
                    )
                    if len(msg2) > 0:
                        msg += msg2
                    if disclosive2:
                        disclosive = True

        return msg, disclosive

    def additional_checks(
        self, curr_separate: dict, saved_separate: dict
    ) -> tuple[str, str]:
        """Placeholder function for additional posthoc checks e.g. keras this
        version just checks that any lists have the same contents"""
        # posthoc checking makes sure that the two dicts have the same set of
        # keys as defined in the list self.examine_separately
        msg = ""
        disclosive = False
        for item in self.examine_seperately_items:
            if isinstance(curr_separate[item], list):
                if len(curr_separate[item]) != len(saved_separate[item]):
                    msg += f"Warning: different counts of values for parameter {item}"
                    disclosive = True
                else:
                    for i in range(len(saved_separate[item])):
                        difference = list(
                            diff(curr_separate[item][i], saved_separate[item][i])
                        )
                        if len(difference) > 0:
                            msg += (
                                f"Warning: at least one non-matching value"
                                f"for parameter list {item}"
                            )
                            disclosive = True
                            break
        return msg, disclosive

    def request_release(self, filename: str = "undefined") -> None:
        """Saves model to filename specified and creates a report for the TRE
        output checkers."""
        if filename == "undefined":
            print("You must provide the name of the file you want to save your model")
            print("For security reasons, this will overwrite previous versions")
        else:
            self.save(filename)
            msg_prel, disclosive_prel = self.preliminary_check(verbose=False)
            msg_post, disclosive_post = self.posthoc_check()
            output: dict = {
                "researcher": self.researcher,
                "model_type": self.model_type,
                "model_save_file": self.model_save_file,
                "details": msg_prel,
            }
            if not disclosive_prel and not disclosive_post:
                output[
                    "recommendation"
                ] = f"Run file {filename} through next step of checking procedure"
            else:
                output["recommendation"] = "Do not allow release"
                output["reason"] = msg_prel + msg_post
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
        self.ignore_items = ["model_save_file", "ignore_items"]
        self.examine_seperately_items = ["tree_"]

    def additional_checks(
        self, curr_separate: dict, saved_separate: dict
    ) -> tuple[str, str]:
        """Decision Tree-specific checks"""
        # call the super function to deal with any items that are lists
        # just in case we add any in the future
        msg, disclosive = super().additional_checks(curr_separate, saved_separate)
        # now deal with the decision-tree specific things
        # which for now means the attribute "tree_" which is a sklearn tree
        for item in self.examine_seperately_items:
            if isinstance(curr_separate[item], Tree):
                # print(f"item {curr_separate[item]} has type {type(curr_separate[item])}")
                diffs_list = list(
                    diff(curr_separate[item].value, saved_separate[item].value)
                )
                if len(diffs_list) > 0:
                    disclosive = True
                    msg += f"structure {item} has  {len(diffs_list)} differences: {diffs_list}"
        return msg, disclosive

    def fit(self, x: np.ndarray, y: np.ndarray):
        """Do fit and then store model dict"""
        super().fit(x, y)
        self.saved_model = copy.deepcopy(self.__dict__)


class SafeRandomForest(SafeModel, RandomForestClassifier):
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
