import itertools, os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
#from tensorflow.keras.models import Model
#from tensorflow.keras.layers import Input, Dense, Dropout
import matplotlib.pyplot as plt
from sklearn.metrics import auc, RocCurveDisplay, DetCurveDisplay, det_curve, confusion_matrix,  classification_report
import scikitplot as skplt
from typing import Any, Iterable, Optional

##############################
####       general        ####
####      functions       ####
##############################

def create_dir(path:str):
    """
    Creates a new directory if it does not exist.

    path: directory to create.
    """
    if not os.path.isdir(path):
        os.mkdir(path)

##############################
#### metrics and plotting ####
####      functions       ####
##############################


def plot_confusion_matrix(name: str, 
                          confusion_matrix: Iterable[float],
                          n_classes:int,
                          normalize: bool = False,
                          cmap:Any = plt.cm.Blues,
                          path:str = os.getcwd()) -> Any:
    """
    This function plots a confusion matrix for predictions of a given model.
    Name: is the name of the model.
    confusion_matrix: is the confusion matrix, e.g. output from confusion_matrix(y_test, y_pred).
    n_classes: is the number of classes, e.g. output of range(2).
    normalize: (boolean) whether to normalize the confusion matrix or not. Default False.
    cmap: is the colormap of the confusion matrix, default is cm.Blues.
    path: it save figure to the given path/figname. Default don't save.
    """
    if normalize:
        confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(confusion_matrix, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    tick_marks = np.arange(n_classes) #np.arange(len(n_classes))
    plt.xticks(tick_marks, range(n_classes), rotation=45)
    plt.yticks(tick_marks, range(n_classes))

    fmt = '.2f' if normalize else 'd'
    thresh = confusion_matrix.max() / 2.
    for i, j in itertools.product(range(confusion_matrix.shape[0]), range(confusion_matrix.shape[1])):
        plt.text(j, i, format(confusion_matrix[i, j], fmt),
               horizontalalignment="center",
               color="white" if confusion_matrix[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title(str(name)+' - Confusion matrix')

    plt.savefig(os.path.join(path, "Confusion_matrix_"+name+".jpeg"), bbox_inches='tight')
    plt.show()


def plotROC_classifier(clf:Any,
                       X_tmp_test: Iterable[float], 
                       y_tmp_test: Iterable[float],
                       title:str = "ROC curve", 
                       path: str = os.getcwd()):
    """
    This function calculates and plots a ROC AUC curve of a given model.

    clf: is the fitted classifier.
    X_tmp_test: test data.
    y_tmp_test: labels of test data.
    title: Set a title for the figure, e.g. specify model name/parameters etc. Default "ROC curve".
    path: it save figure to the given path/figname. Default don't save.
    """
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    fig, ax = plt.subplots()
    viz = RocCurveDisplay.from_estimator(
            clf,
            X_tmp_test,
            y_tmp_test,
            name="ROC fold {}".format(0),
            alpha=0.3,
            lw=1,
            ax=ax,
        )
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(viz.roc_auc)

    ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(
        mean_fpr,
        mean_tpr,
        color="b",
        label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
        lw=2,
        alpha=0.8,
    )

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color="grey",
        alpha=0.2,
        label=r"$\pm$ 1 std. dev.",
    )

    ax.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
        title=title,
    )
    ax.legend(loc="lower right")

    #if save:
    fig.savefig(os.path.join(path, title+".jpeg"), bbox_inches='tight')
    #else:
    plt.show()
    return mean_auc



def plot_prob_test_train(pred_test: Iterable[float], 
                         pred_train: Iterable[float], 
                         title: str ='Membership probalibilty',
                         plot_class: int = 0,
                         path: str = os.getcwd()):
    """
    This function plots and histogram of the probability associated
    with MIA (memebership inference attack).

    pred_test: predicted probability of the test data
    (from the MIA split, not from building the model).
    pred_train: predicted probability of the train data
    (from the MIA split, not from building the model).
    title: Set a title for the figure, e.g. specify model name/parameters
    and MIA etc. Default "Membership probalibilty".
    plot_class: which class to plot the probabilities for. Class number is apended to the title
    path: it save figure to the given path/figname. Default don't save.
    """
    plt.hist(np.array(pred_train)[:, plot_class],  alpha=0.5, bins=20, label='Training Data (Members)',
                histtype='bar', range=(0, 1))
    plt.hist(np.array(pred_test)[:, plot_class],  alpha=0.5, bins=20, label='Test Data (Non-members)',
                histtype='bar', range=(0, 1))
    plt.legend(loc='center left', bbox_to_anchor=(0.5, -0.25))
    plt.xlabel('Membership Probability')
    plt.ylabel('Fraction')
    plt.title(f'{title} (class {plot_class})')

    plt.savefig(os.path.join(path, f'{title}_class_{plot_class}.jpeg'), bbox_inches='tight')

    plt.show()

    
def plot_detection_error_tradeoff(clf:Any,
                                  X_test:Iterable[float], 
                                  y_test:Iterable[float],
                                  title:str = "Detection Error Tradeoff (DET) curve", 
                                  model_name:Optional[str] = None,
                                  path:str = os.getcwd()):
    """
    Plot the Detection Error Traeoff (DET) according to the definition in
    https://scikit-learn.org/stable/modules/model_evaluation.html#det-curve
    
    clf: is the fitted classifier.
    X_test: test data.
    y_test: labels of test data.
    title: Set a title for the figure, e.g. specify model name/parameters etc. Default "ROC curve".
    model_name: is the name of the model.
    path: it save figure to the given path/figname. Default don't save.
    """
    
    print("WARNING: when the false negative rate or false positive rate is 0, no line will appear in the plot.")
    
    DetCurveDisplay.from_estimator(clf, X_test, y_test, response_method='auto', name=model_name)#'predict_proba'

    plt.title(title)
    plt.grid(linestyle="--")
    plt.legend()
    plt.savefig(os.path.join(path, title+"_"+model_name+".jpeg"), bbox_inches='tight')
    plt.show()
    
    
#def attacker_advantage(clf,
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


def plot_ks_metric(y_test:Iterable[float],
                   y_probas:Iterable[float],
                   title:str = "KS statistic plot", 
                   path:str = os.getcwd()):
    """
    This functions plots the KS statistic.
    y_test: labels for the data
    y_probas: predicted probabilities
    """
    skplt.metrics.plot_ks_statistic(y_test, y_probas)
    plt.title(title)
    plt.savefig(os.path.join(path, title+".jpeg"), bbox_inches='tight')
    plt.show()
    
    
def plot_calibration_curve(y_test:Iterable,
                           clf_names:Iterable[str],
                           #clf_list:Iterable[Any],
                           probas_list:Iterable[float],
                           path:str = os.getcwd()):
    """
    Plots the calibration curves of several fitted models.
    
    y_test: list of labels
    clf_names: list of classifiers
    proba_list: list of probabilities predicted by the classfiers (same order as clf_names)
    """

    skplt.metrics.plot_calibration_curve(y_test, probas_list, clf_names)
    plt.savefig(os.path.join(path, "calibration_curves.jpeg"), bbox_inches='tight')
    plt.show()

##############################
####    Attack related    ####
####      functions       ####
##############################

def split_target_data(X:Iterable[float],
                      y:Iterable,
                     i:Optional[int]=0.5,
                     j:Optional[int]=0.5):
    """
    This functions splits the data into 3 sets: target, train and test.
    
    X: matrix of feature data
    y: labels of the data
    i: size of the target vs train+test data
    j: size of train vs test data
    """
    X_target, X_test_tmp, y_target, y_test_tmp = train_test_split(X, y, shuffle=True, test_size=i, random_state=58954)
    X_train, X_test, y_train, y_test = train_test_split(X_test_tmp, y_test_tmp, shuffle=True, test_size=j, random_state=58954)
    return(X_target, X_train, X_test, y_target, y_train, y_test)


def calculate_metrics_plots(model:Any,
                            name:str,
                            X_test:Iterable[float],
                            y_test:Iterable,
                            path:str):
    """
    Calculate metrics and plot AUC, confusion matrix.
    Returns the predicted probabilities.
    
    model: the fitted classifier
    name: name of the model
    X_test: data to obtain posteriors
    y_test: labels of the data
    path: path where to save images
    """
    create_dir(path)
    y_pred = model.predict(X_test)
    print(classification_report(y_test,y_pred))
    metrics = get_metrics(model, X_test, y_test)
    print('Metrics')
    [print(k,v) for k,v in metrics.items()]
    plot_confusion_matrix(name, confusion_matrix(y_test, y_pred), 2, path)
    metrics['AUC'] = plotROC_classifier(model, X_test, y_test, name, path)
    
    return(model.predict_proba(X_test), metrics)

def create_mia_data(clf:Any, 
                    xtrain:Iterable[float], 
                    xtest:Iterable[float], 
                    sort_probs:bool=False,
                    keep_top:int=-1):
    """
    This function predict the probability of train and test data
    (from building the model split) belonging to their
    corresponding train or test respectively
    and it also creates labels.

    clf: fitted classifier (original model)
    xtrain: training data (original model)
    xtest: test data (original model)
    """    
    miX = np.concatenate(
        (
            clf.predict_proba(xtrain),
            clf.predict_proba(xtest)
        )
    )
    
    if sort_probs:
        miX = -np.sort(-miX, axis=1)
        if keep_top > -1:
            miX = miX[:, :keep_top]

    miY = np.concatenate(
        (
        np.ones((len(xtrain), 1), int),
        np.zeros((len(xtest), 1), int)
        )
    ).flatten()
    return(miX,miY)


def run_membership_inference_attack(clf_name:str,
                                    model:Any, 
                                    xtrain:Iterable[float], 
                                    xtest:Iterable[float], 
                                    MIA_classifier:Any, 
                                    MIA_name:str, 
                                    path):
    """
    This function performs MIA (membership inference attack)
    and plots metrics.

    clf_name: name of the original model in which MIA is performed.
    model: original model in which MIA is performed.
    xtrain: train data of the original model.
    xtest: test data of the original model.
    MIA_classifier: unfitted classifier for MIA.
    MIA_name: name of the MIA classifier.
    path: path where to save the generated images.
    """

    miX, miY = create_mia_data(model, xtrain, xtest, sort_probs=True)
    mi_train_x, mi_test_x, mi_train_y, mi_test_y = train_test_split(miX, miY, test_size=0.2, stratify=miY)

    mi_rf = MIA_classifier
    mi_rf.fit(mi_train_x, mi_train_y)

    auc = plotROC_classifier(mi_rf, mi_test_x, mi_test_y, f'{clf_name} - MIA {MIA_name}', path)

    pred_y = model.predict_proba(xtest)
    pred_train_y = model.predict_proba(xtrain)

    _, n_classes = pred_y.shape
    for cl in range(n_classes):
        plot_prob_test_train(pred_y, pred_train_y, f'{clf_name} - MIA {MIA_name}', cl, path)
    
    plot_detection_error_tradeoff(mi_rf, mi_test_x, mi_test_y, clf_name+" Detection Error Tradeoff (DET) curve", clf_name, path)
    
    print('Metrics')
    metrics = get_metrics(mi_rf, mi_test_x, mi_test_y)
    metrics['AUC'] = auc
    [print(k,v) for k,v in metrics.items()]
    
    proba = np.concatenate((pred_y,pred_train_y))
    plot_ks_metric(miY, proba, "MIA "+clf_name+" KS statistic plot", path)
    
    return((pred_y,pred_train_y), miY, metrics)


def mia_worst_case(name: str,
                  model:Any,
                  X_train:Iterable[float],
                  X_test:Iterable):
    """
    Function to perform worst case scenario attack. This consisits 
    of the attacker knows the target data, the model and parameters.
    
    The attacker has access to the data used to train the model and a second dataset from the same distribution (e.g. the original data is split into training and test portions and the attacker has access to both). They attempt to develop a predictor of whether a sample is in the training set using only the probabilities outputted from the target model. This can be thought of as an attacker interal to the TRE who plans to leak a model which can be used for MIAs outside it.
    
    name: classifier name
    model: fitted classifier
    X_train: training data
    X_test: test data
    
    return MIA probabilies, MIA y_labels and metrics.
    """
    path = "worst_case"
    create_dir(path)
    return(run_membership_inference_attack(name, model, X_train, X_test, RandomForestClassifier(), 'RandomForest', path))


def  mia_salem_adversary(shadow_clf: Any,
                        shadow_clf_name: str,
                        X_shadow_train:Iterable[float],
                        y_shadow_train:Iterable,
                        X_test: Iterable[float],
                        y_test: Iterable,
                        attack_type: str):
    """
    This function is desinged as a setup for adversary attacks defined in Salem adversary 1 and 2.
    
    shadow_clf: classifier for shadow model (not fitted)
    shadow_clf_name: shadow classifier name
    X_shadow_train: training data
    y_shadow_train: training data labels
    X_test: test data
    y_test: test data labels
    attack_type: type of attack, e.g. Salem1 or Salem2
    """
    create_dir(attack_type)
    
    #model_metrics = {}
    #1 shadow model training
    shadow_model = shadow_clf
    shadow_model.fit(X_shadow_train, y_shadow_train)
    model_metrics = get_metrics(shadow_model, X_test, y_test)
    model_metrics['AUC'] = plotROC_classifier(shadow_model, X_test, y_test, f'{shadow_clf_name} - Shadow model', attack_type)
    
    #2 attack model training
    # Get prediction probabilities from the shadow model    
    mia_name = 'RandomForest'
    proba, y_labels, mia_metrics = run_membership_inference_attack(shadow_clf_name,
shadow_model, X_shadow_train,X_test, RandomForestClassifier(), 'RandomForest', attack_type)
    return(model_metrics, proba, y_labels, mia_metrics)


def mia_salem_1(shadow_clf: Any,
                shadow_clf_name: str,
                X_shadow_train:Iterable[float],
                y_shadow_train:Iterable,
                X_test: Iterable[float],
                y_test: Iterable):
    """
    Perform Salem adversary 1 type of attack. This attack assumes attacker has a dataset of the same distribution as the target's model tarining data. The shadow model mimic the target's model behaviour.

    
    shadow_clf: classifier for shadow model (not fitted)
    shadow_clf_name: shadow classifier name
    X_shadow_train: shadow model training data, which must be of the same distribution of the target data, e.g. could be a split of the one used for training/test the target model and not used to train the target.
    y_shadow_train: training data labels
    X_test: test data
    y_test: test data labels
    attack: type of attack, e.g. Salem1 or Salem2
    """
    create_dir("Salem1")
    return(mia_salem_adversary(shadow_clf, shadow_clf_name, X_shadow_train, y_shadow_train, X_test, y_test, "Salem1"))


def mia_salem_2(shadow_clf: Any,
                shadow_clf_name: str,
                X_shadow_train:Optional=None,
                y_shadow_train:Optional=None,
                X_test: Optional=None,
                y_test: Optional=None):
    """
    Perform Salem adversary 2 type of attack. This attack assumes attacker does not have a dataset of the same distribution as the target's model tarining data. The adversary does not know the structure of the target mode.

    
    shadow_clf: classifier for shadow model (not fitted)
    shadow_clf_name: shadow classifier name
    X_shadow_train: shadow model training data, which must be of the same distribution of the target data, e.g. could be a split of the one used for training/test the target model and not used to train the target. Default: breast cancer data provided with sklearn datasets.
    y_shadow_train: training data labels. Default: labels breast cancer data provided with sklearn datasets.
    X_test: test data. Default: split of breast cancer data provided with sklearn datasets.
    y_test: test data labels. Default: labels from a split of breast cancer data provided with sklearn datasets.
    attack: type of attack, e.g. Salem1 or Salem2
    """
    create_dir("Salem2")
    if not X_shadow_train:
        X_breast_cancer, y_breast_cancer = datasets.load_breast_cancer(return_X_y=True)
        X_shadow_train, X_test, y_shadow_train, y_test = train_test_split(X_breast_cancer, y_breast_cancer, shuffle=True, test_size=0.3, random_state=58954)
    
    return(mia_salem_adversary(shadow_clf, shadow_clf_name, X_shadow_train, y_shadow_train, X_test, y_test, "Salem2"))