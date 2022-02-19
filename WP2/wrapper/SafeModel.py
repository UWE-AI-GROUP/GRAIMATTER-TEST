"""This module contains prototypes of privacy safe model wrappers."""

from __future__ import annotations

import getpass
import json
import pickle
from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


def check_min(key: str, val: Any, cur_val: Any) -> tuple[str, bool]:
    """Checks minimum value constraint."""
    if cur_val > val:
        possibly_disclosive = False
        msg = (
            f"- parameter {key} increased"
            f" from recommended min value of {val} to {cur_val}."
            " This is not problematic.\n"
        )
    else:
        possibly_disclosive = True
        msg = (
            f"- parameter {key} decreased"
            f" from recommended min value of {val} to {cur_val}."
            " THIS IS POTENTIALLY PROBLEMATIC.\n"
        )
    return msg, possibly_disclosive


def check_max(key: str, val: Any, cur_val: Any) -> tuple[str, bool]:
    """Checks maximum value constraint."""
    if cur_val < val:
        possibly_disclosive = False
        msg = (
            f"- parameter {key} decreased"
            f" from recommended max value of {val} to {cur_val}."
            " This is not problematic.\n"
        )
    else:
        possibly_disclosive = True
        msg = (
            f"- parameter {key} increased"
            f" from recommended max value of {val} to {cur_val}."
            " THIS IS POTENTIALLY PROBLEMATIC.\n"
        )
    return msg, possibly_disclosive


def check_equal(key: str, val: Any, cur_val: Any) -> tuple[str, bool]:
    """Checks equality value constraint."""
    if cur_val == val:
        possibly_disclosive = False
        msg = (
            f"- parameter {key} changed"
            f" from value of {val} to recommended {cur_val}."
            " This is not problematic.\n"
        )
    else:
        possibly_disclosive = True
        msg = (
            f"- parameter {key} changed"
            f" from recommended fixed value of {val} to {cur_val}."
            " THIS IS POTENTIALLY PROBLEMATIC.\n"
        )
    return msg, possibly_disclosive


class SafeModel:
    """Privacy protected model base class."""

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
                pickle.dump(self.model, file)
        else:
            print("only .pkl file saves currently implemented")

    def get_constraints(self) -> dict:
        """Gets constraints relevant to the model type from the master read-only file."""
        rules: dict = {}
        with open("rules.json", "r", encoding="utf-8") as json_file:
            parsed = json.load(json_file)
            rules = parsed[self.model_type]
        return rules["rules"]

    def __check_model_param(
        self, rule: dict, apply_constraints: bool
    ) -> tuple[str, bool]:
        """Checks whether current model parameter has been changed from
        constrained settings. Optionally fixes violations."""
        op_msg: str = ""
        op_disclosive: bool = False
        operator: str = rule["operator"]
        key: str = rule["keyword"]
        val = rule["value"]
        cur_val = getattr(self, key)
        if cur_val == val:
            op_msg = f"- parameter {key} unchanged at recommended value {val}"
        elif operator == "min":
            op_msg, op_disclosive = check_min(key, val, cur_val)
        elif operator == "max":
            op_msg, op_disclosive = check_max(key, val, cur_val)
        elif operator == "equals":
            op_msg, op_disclosive = check_equal(key, val, cur_val)
        else:
            op_msg = f"- unknown operator in parameter specification {operator}"
        if apply_constraints and op_disclosive:
            setattr(self, key, val)
        return op_msg, op_disclosive

    def check_model_params(self, apply_constraints: bool = False) -> tuple[str, bool]:
        """Checks whether current model parameters have been changed from
        constrained settings. Automatically fixes violated constraints."""
        possibly_disclosive: bool = False
        msg: str = ""
        rules: dict = self.get_constraints()
        for rule in rules:
            op_disclosive = False
            op_msg = ""
            operator = rule["operator"]
            if operator == "and":  # shallow sub rules
                for arg in rule["subexpr"]:
                    sop_msg, sop_disclosive = self.__check_model_param(
                        arg, apply_constraints
                    )
                    op_msg += sop_msg
                    if sop_disclosive:
                        op_disclosive = True
            elif operator == "or":  # no automatic fixing
                for arg in rule["subexpr"]:
                    op_disclosive = True
                    sop_msg, sop_disclosive = self.__check_model_param(arg, False)
                    op_msg += sop_msg
                    if not sop_disclosive:
                        op_disclosive = False
            else:
                op_msg, op_disclosive = self.__check_model_param(
                    rule, apply_constraints
                )
            if op_disclosive:
                possibly_disclosive = True
            msg += op_msg + "\n"
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
                        f"RECOMMENDATION: Do not allow release of file {filename}\n\n"
                    )
                else:
                    file.write(
                        "Model has not been changed to increase risk of disclosure:\n"
                        f"{msg}\n"
                        "RECOMMENDATION: "
                        f"Run file {filename} through next step of checking procedure\n\n"
                    )

    def preliminary_check(self) -> tuple[str, bool]:
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
        return msg, possibly_disclosive

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
        super().check_model_params(apply_constraints=True)


class SafeRandomForest(SafeModel, RandomForestClassifier):
    """Privacy protected Random Forest classifier."""

    def __init__(self, **kwargs: Any) -> None:
        """Creates model and applies constraints to params"""
        SafeModel.__init__(self)
        RandomForestClassifier.__init__(self, **kwargs)
        self.model_type: str = "RandomForestClassifier"
        super().check_model_params(apply_constraints=True)

    # def __getattr__(self, attr):
    #    if attr in self.__dict__:
    #        return getattr(self, attr)
    #    return getattr(self, attr)
