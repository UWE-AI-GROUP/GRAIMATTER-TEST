from sklearn.metrics import confusion_matrix
from typing import Any, Iterable, Optional
from sklearn.metrics import auc, roc_curve


def get_metrics(clf,
                X_test:Iterable[float], 
                y_test:Iterable[float]):
    """
    Calculate metrics, including attacker advantage for MIA binary.
    Implemented as Definition 4 on https://arxiv.org/pdf/1709.01604.pdf
    which is also implemented in tensorFlow-privacy https://github.com/tensorflow/privacy
    
    clf: fitted model.
    X_test: test data.
    y_test: test data labels.
    
    returns a dictionary with several metrics.
    
    True positive rate or recall (TPR)
    False positive rate (FPR), proportion of negative examples incorrectly classified as positives
    False alarm rate (FAR), proportion of objects classified as positives that are incorrect, also known as false discovery rate
    True neagative rate (TNR)
    Positive predictive value or precision (PPV)
    Negative predictive value (NPV)
    False neagative rate (FNR)
    Accuracy (ACC)
    Advantage
    Positive likelihood ratio (PLR)
    Negative likelihood ratio (NLR)
    Odds ratio (OR)
    """
    metrics = {}
    y_pred = clf.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    metrics['TPR'] = tp / (tp + fn) #true positive rate or recall
    metrics['FPR'] = fp / (fp + tn) #false positive rate, proportion of negative examples incorrectly classified as positives
    metrics['FAR'] = fp / (fp + tp) #proportion of things classified as positives that are incorrect, also known as false discovery rate
    metrics['TNR'] = tn / (tn + fp) #true negative rate or specificity
    metrics['PPV'] = tp / (tp + fp) #precision or positive predictive value
    metrics['NPV'] = tn / (tn + fn) #negative predictive value
    metrics['FNR'] = fn / (tp + fn) #false negative rate
    metrics['ACC'] = (tp + tn) / (tp + fp + fn + tn) #overall accuracy
    metrics['Advantage'] = abs(metrics['TPR']-metrics['FPR'])
    metrics['PLR'] = metrics['TPR'] / metrics['FPR'] #positive likelihood ratio
    metrics['NLR'] = metrics['FNR'] / metrics['TNR'] #negative likelihood ratio
    metrics['OR'] = metrics['PLR'] / metrics['NLR'] #odds ratio, the odds ratio is used to find the probability of an outcome of an event when there are two possible outcomes
    return metrics
