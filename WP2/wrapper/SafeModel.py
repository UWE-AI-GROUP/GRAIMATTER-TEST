"""This module contains prototypes of privacy safe model wrappers."""

from __future__ import annotations

import getpass
import pickle
from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


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

    def apply_constraints(self, **kwargs: Any) -> None:
        """Sets model attributes according to constraints."""
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
        constrained settings."""
        possibly_disclosive: bool = False
        msg: str = ""
        params: dict = self.get_constraints()
        for key, (operator, recommended_val) in params.items():
            current_val = str(eval(f"self.{key}"))
            if current_val == recommended_val:
                msg += f"- parameter {key} unchanged at recommended value {recommended_val}"
            elif operator == "min":
                if float(current_val) > float(recommended_val):
                    msg += (
                        f"- parameter {key} increased"
                        f" from recommended min value of {recommended_val} to {current_val}."
                        " This is not problematic.\n"
                    )
                else:
                    possibly_disclosive = True
                    msg += (
                        f"- parameter {key} decreased"
                        f" from recommended min value of {recommended_val} to {current_val}."
                        " THIS IS POTENTIALLY PROBLEMATIC.\n"
                    )
            elif operator == "max":
                if float(current_val) < float(recommended_val):
                    msg += (
                        f"- parameter {key} decreased"
                        f" from recommended max value of {recommended_val} to {current_val}."
                        " This is not problematic.\n"
                    )
                else:
                    possibly_disclosive = True
                    msg += (
                        f"- parameter {key} increased"
                        f" from recommended max value of {recommended_val} to {current_val}."
                        " THIS IS POTENTIALLY PROBLEMATIC.\n"
                    )
            elif operator == "equals":
                possibly_disclosive = True
                msg += (
                    f"- parameter {key} changed"
                    f" from recommended fixed value of {recommended_val} to {current_val}."
                    " THIS IS POTENTIALLY PROBLEMATIC.\n"
                )
            else:
                msg += f"- unknown operator in parameter specification {operator}"
            msg += "\n"
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
        super().apply_constraints(**kwargs)


class SafeRandomForest(SafeModel, RandomForestClassifier):
    """Privacy protected Random Forest classifier."""

    def __init__(self, **kwargs: Any) -> None:
        """Creates model and applies constraints to params"""
        SafeModel.__init__(self)
        RandomForestClassifier.__init__(self, **kwargs)
        self.model_type: str = "RandomForestClassifier"
        super().apply_constraints(**kwargs)

    # def __getattr__(self, attr):
    #    if attr in self.__dict__:
    #        return getattr(self, attr)
    #    return getattr(self, attr)
