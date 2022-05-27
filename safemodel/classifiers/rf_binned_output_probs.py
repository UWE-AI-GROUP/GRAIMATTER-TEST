'''
A Random forest with discretised output probabilities
'''
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def bin_probabilities(probs: np.ndarray, n_probability_bins: int) -> np.ndarray:
    '''Method to bin labels'''
    # Create the bins e.g. if self.n_probability_bins = 10 then the bins will be
    # (-0.01, 0.05]
    # (0.05, 0.15]
    # (0.15, 0.25]
    # ....
    # (0.95, 1.01]
    # Note we start at -0.01 anmd finish at 1.01 to ensure we don't miss exact zero values

    start_bin = 1 / (n_probability_bins * 2)
    end_bin = 1 - start_bin
    bins = np.linspace(start_bin, end_bin, n_probability_bins)
    bins = [-0.01] + list(bins) + [1.01]
    labels = np.linspace(0., 1., n_probability_bins + 1)
    binned_probs = np.zeros_like(probs, float)
    _, n_probs = probs.shape
    # Do the binning with pandas cut method
    for m in range(n_probs):
        binned_probs[:, m] = pd.cut(probs[:, m], bins, labels=labels)

    # Re-normalise, just to be safe
    binned_probs /= binned_probs.sum(axis=1)[:, None]
    return(binned_probs)

class RFBinnedOutput(RandomForestClassifier):
    '''
    Class to 
    '''
    def __init__(self, n_probability_bins=10, **kwargs):
        super().__init__(**kwargs)
        self.n_probability_bins = n_probability_bins

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        '''Override of predict_proba to bin probs'''
        # Get normal predictions
        probs = super().predict_proba(X)
        if self.n_probability_bins == 0: # When 0, revert to normal behaviour
            return probs
        else:
            return bin_probabilities(probs, self.n_probability_bins)



if __name__ == '__main__':
    rr = RFBinnedOutput(min_samples_split=2, n_probability_bins=5)
    from data_preprocessing.data_interface import get_data_sklearn
    X, y = get_data_sklearn('mimic2-iaccd')
    rr.fit(X.values, y.values.flatten())

    probs = rr.predict_proba(X.values)
    print(probs)
