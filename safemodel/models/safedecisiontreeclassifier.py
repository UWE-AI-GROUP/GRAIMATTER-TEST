class SafeDecisionTreeClassifier(SafeModel, DecisionTreeClassifier):
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
                    msg += f"structure {item} has {len(diffs_list)} differences: {diffs_list}"
        return msg, disclosive

    def fit(self, x: np.ndarray, y: np.ndarray):
        """Do fit and then store model dict"""
        super().fit(x, y)
        self.saved_model = copy.deepcopy(self.__dict__)
