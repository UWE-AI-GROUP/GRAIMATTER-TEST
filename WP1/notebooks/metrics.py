import sys
from sklearn.metrics import confusion_matrix
from typing import Any, Iterable, Optional
from sklearn.metrics import roc_auc_score

def div(x,y, default):
    if y!=0:
        return round(float(x/y),8)
    else:
        #print('Warning: division by 0', x,y)
        return float(default)


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
    #print('tn', tn, 'fp',fp,'fn', fn,'tp', tp)
    
    metrics['TPR'] = round(float(tp/(tp + fn)), 8) #true positive rate or recall
    metrics['FPR'] = round(float(fp / (fp + tn)), 8) #false positive rate, proportion of negative examples incorrectly classified as positives
    metrics['FAR'] = div(fp, (fp + tp), 0) #False alarm rate, proportion of things classified as positives that are incorrect, also known as false discovery rate
    metrics['TNR'] = round(float(tn / (tn + fp)), 8) #true negative rate or specificity
    metrics['PPV'] = div(tp, (tp + fp), 0) #precision or positive predictive value
    metrics['NPV'] = div(tn, (tn + fn), 0) #negative predictive value
    metrics['FNR'] = round(float(fn / (tp + fn)), 8) #false negative rate
    metrics['ACC'] = round(float((tp + tn) / (tp + fp + fn + tn)), 8) #overall accuracy
    metrics['F1score'] = round(float(2*((metrics['PPV']*metrics['TPR']) / (metrics['PPV']+metrics['TPR']), 0)))#harmonic mean of precision and sensitivity
    metrics['Advantage'] = float(abs(metrics['TPR']-metrics['FPR']))
    #metrics['PLR'] = float(metrics['TPR'] / metrics['FPR']) #positive likelihood ratio
    #metrics['NLR'] = float(metrics['FNR'] / metrics['TNR']) #negative likelihood ratio
    #metrics['OR'] = metrics['PLR'] / metrics['NLR'] #odds ratio, the odds ratio is used to find the probability of an outcome of an event when there are two possible outcomes
    #calculate AUC of model
    y_pred_proba = clf.predict_proba(X_test)[::,1]
    metrics['AUC'] = round(roc_auc_score(y_test, y_pred_proba),8)
    
    
    return metrics