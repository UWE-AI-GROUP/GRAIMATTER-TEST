from dp_svc import DPSVC

class SafeSVC(SafeModel, DPSVC):

    def __init__(self, C=1., gamma='scale', dhat=1000, eps=10, **kwargs):
        SafeModel.__init__(self)
        DPSVC.__init__(self)
        self.model_type: str = "SVC"
        self.ignore_items = [
            "model_save_file",
            "ignore_items",
            "train_features",
            "train_labels",
            "unique_labels",
            "train_labels",
            "weights",
            "noisy_weights"
            
        ]

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """Do fit and then store model dict"""
        super().fit(x, y)
        self.saved_model = copy.deepcopy(self.__dict__)

