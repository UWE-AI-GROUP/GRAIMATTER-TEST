'''
Likelihood testing scenario from https://arxiv.org/pdf/2112.03570.pdf
'''
# pylint: disable = invalid-name
from socket import NI_DGRAM
from typing import Iterable
import numpy as np
from scipy.stats import norm

from attacks.scenarios import worst_case_mia


EPS = 1e-16

class DummyClassifier:
    def __init__(self, y_probs):
        self.probs = y_probs
        print(self.probs.shape)
    
    def predict(self, test_X):
        return 1 * (self.probs[:, 1] > 0.5)
    
    def predict_proba(self, test_X):
        return self.probs

def logit(p: float) -> float:
    '''Standard logit function'''
    if p > 1 - EPS:
        p = 1 - EPS
    return np.log(p / (1 - p))


def likelihood_scenario(
    target_model,
    shadow_clf,
    X_target_train: Iterable[float],
    y_target_train: Iterable[float],
    X_shadow_train: Iterable[float],
    y_shadow_train: Iterable[float],
    X_test: Iterable[float],
    prop_mia_train: float=0.5
):
    '''
    Implements the likelihood test, using the "offline" version
    See p.6 (top of second column) for details
    '''
    N_SHADOW_MODELS = 100
    n_train_rows, _ = X_target_train.shape
    n_shadow_rows, _ = X_shadow_train.shape
    indices = np.arange(0, n_train_rows + n_shadow_rows, 1)

    # Combine taregt and shadow train, from which to sample datasets
    combined_X_train = np.vstack((
        X_target_train,
        X_shadow_train
    ))
    combined_y_train = np.hstack((
        y_target_train,
        y_shadow_train
    ))

    train_row_to_confidence = {i: [] for i in range(n_train_rows)}
    shadow_row_to_confidence = {i: [] for i in range(n_shadow_rows)}
    
    # Train N_SHADOW_MODELS shadow models
    for model_idx in range(N_SHADOW_MODELS):
        if model_idx % 100 == 0:
            print(f'{model_idx}/{N_SHADOW_MODELS}')

        # Pick the indices to use for training this one
        these_idx = np.random.choice(indices, n_train_rows, replace=False)
        temp_X_train = combined_X_train[these_idx, :]
        temp_y_train = combined_y_train[these_idx]

        # Fit the shadow model
        shadow_clf.fit(temp_X_train, temp_y_train)

        # Get the predicted probabilities on the training data
        confidences = shadow_clf.predict_proba(X_target_train)
        these_idx = set(these_idx)
        for i in range(n_train_rows):
            if i not in these_idx:
                # If i was _not_ used for training, incorporate the logit of its confidence of
                # being correct - TODO: should we just be taking max??
                train_row_to_confidence[i].append(
                    logit(
                        confidences[i, y_target_train[i]]
                    )
                )
        # Same process for shadow data
        shadow_confidences = shadow_clf.predict_proba(X_shadow_train)
        for i in range(n_shadow_rows):
            if i + n_train_rows not in these_idx:
                shadow_row_to_confidence[i].append(
                    logit(
                        shadow_confidences[i, y_shadow_train[i]]
                    )
                )
    
    # Compute predictive probabilities on train and shadow data for the _target_ model
    target_train_preds = target_model.predict_proba(X_target_train)
    shadow_train_preds = target_model.predict_proba(X_shadow_train)

    # Do the test described in the paper in each case
    mia_scores = []
    mia_labels = []
    for i in range(n_train_rows):
        true_score = logit(target_train_preds[i, y_target_train[i]])
        null_scores = np.array(train_row_to_confidence[i])
        mean_null = null_scores.mean()
        var_null = null_scores.var()
        prob = norm.cdf(true_score, loc=mean_null, scale=np.sqrt(var_null))
        mia_scores.append([1 - prob, prob])
        mia_labels.append(1)
    
    for i in range(n_shadow_rows):
        true_score = logit(shadow_train_preds[i, y_shadow_train[i]])
        null_scores = np.array(shadow_row_to_confidence[i])
        mean_null = null_scores.mean()
        var_null = null_scores.var()
        prob = norm.cdf(true_score, loc=mean_null, scale=np.sqrt(var_null))
        mia_scores.append([1 - prob, prob])
        mia_labels.append(0)

    mia_clf = DummyClassifier(np.array(mia_scores))

    return np.array(mia_scores), np.array(mia_labels), mia_clf


def main():
    from data_preprocessing.data_interface import get_data_sklearn
    from attacks.scenarios import split_target_data
    from sklearn.ensemble import RandomForestClassifier
    from attacks.metrics import get_metrics
    X, y = get_data_sklearn('mimic2-iaccd')

    X_target, X_train, X_test, y_target, y_train, y_test = split_target_data(
        X.values,
        y.values.flatten()
    )

    rf = RandomForestClassifier(min_samples_leaf=1, min_samples_split=2, max_depth=10)
    rf.fit(X_target, y_target)
    mia_test_probs, mia_test_labels, mia_clf = likelihood_scenario(rf, RandomForestClassifier(min_samples_leaf=1, min_samples_split=2, max_depth=10), X_target, y_target, X_train, y_train, X_test)

    metrics = get_metrics(mia_clf, mia_test_probs, mia_test_labels)
    print(metrics)

    import pylab as plt
    pos = np.where(mia_test_labels == 1)[0]
    plt.subplots(1, 2, 1)
    plt.hist(mia_test_probs[pos, 1])
    pos = np.where(mia_test_labels == 0)[0]
    plt.subplot(1, 2, 2)
    plt.hist(mia_test_probs[pos, 0])
    plt.show()

    mia_test_probs, mia_test_labels, mia_clf = worst_case_mia(rf, X_target, X_test)
    metrics = get_metrics(mia_clf, mia_test_probs, mia_test_labels)
    print(metrics)

if __name__ == '__main__':
    main()