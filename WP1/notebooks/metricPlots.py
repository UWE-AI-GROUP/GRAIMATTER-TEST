import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import auc, RocCurveDisplay, DetCurveDisplay, det_curve, confusion_matrix
import numpy as np
from typing import Any, Iterable, Optional


def plot_confusion_matrix(name: str, 
                          confusion_matrix: Iterable[float],
                          n_classes:int,
                          normalize: bool = False,
                          cmap:Any = plt.cm.Blues,
                          save: Optional[str] = None) -> Any:
    """
    This function plots a confusion matrix for predictions of a given model.
    Name: is the name of the model.
    confusion_matrix: is the confusion matrix, e.g. output from confusion_matrix(y_test, y_pred).
    n_classes: is the number of classes, e.g. output of range(2).
    normalize: (boolean) whether to normalize the confusion matrix or not. Default False.
    cmap: is the colormap of the confusion matrix, default is cm.Blues.
    save: it save figure to the given path/figname. Default don't save.
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

    if save:
        plt.savefig(save, bbox_inches='tight')
    #else:
    plt.show()


def plotROC_classifier(clf:Any,
                       X_tmp_test: Iterable[float], 
                       y_tmp_test: Iterable[float],
                       title:str = "ROC curve", 
                       save: Optional[str] = None):
    """
    This function calculates and plots a ROC AUC curve of a given model.

    clf: is the fitted classifier.
    X_tmp_test: test data.
    y_tmp_test: labels of test data.
    title: Set a title for the figure, e.g. specify model name/parameters etc. Default "ROC curve".
    save: it save figure to the given path/figname. Default don't save.
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

    if save:
        fig.savefig(save, bbox_inches='tight')
    #else:
    plt.show()



def plot_prob_test_train(pred_test: Iterable[float], 
                         pred_train: Iterable[float], 
                         title: str ='Membership probalibilty',
                         plot_class: int = 0,
                         save: Optional[str] = None):
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
    save: it save figure to the given path/figname. Default don't save.
    """
    #fig, ax = plt.subplots()
    plt.hist(np.array(pred_train)[:, plot_class],  alpha=0.5, bins=20, label='Training Data (Members)',
                histtype='bar', range=(0, 1))
    plt.hist(np.array(pred_test)[:, plot_class],  alpha=0.5, bins=20, label='Test Data (Non-members)',
                histtype='bar', range=(0, 1))
    plt.legend(loc='center left', bbox_to_anchor=(0.5, -0.25))
    plt.xlabel('Membership Probability')
    plt.ylabel('Fraction')
    plt.title(f'{title} (class {plot_class})')

    if save:
        plt.savefig(save, bbox_inches='tight')
    #else:
    plt.show()

    
def plot_detection_error_tradeoff(clf:Any,
                                  X_test:Iterable[float], 
                                  y_test:Iterable[float],
                                  title:str = "Detection Error Tradeoff (DET) curve", 
                                  model_name:Optional[str] = None,
                                  save:Optional[str] = None):
    """
    Plot the Detection Error Traeoff (DET) according to the definition in
    https://scikit-learn.org/stable/modules/model_evaluation.html#det-curve
    
    clf: is the fitted classifier.
    X_test: test data.
    y_test: labels of test data.
    title: Set a title for the figure, e.g. specify model name/parameters etc. Default "ROC curve".
    model_name: is the name of the model.
    save: it save figure to the given path/figname. Default don't save.
    """
    
    DetCurveDisplay.from_estimator(clf, X_test, y_test, response_method='predict_proba', name=model_name)

    plt.title(title)
    plt.grid(linestyle="--")
    plt.legend()
    if save:
        plt.savefig(save, bbox_inches='tight')
    plt.show()
    
    
def attacker_advantage(clf,
                       X_test:Iterable[float], 
                       y_test:Iterable[float]):
    """
    Calculate attacker advantage for MIA binary.
    Implemented as Definition 4 on https://arxiv.org/pdf/1709.01604.pdf
    which is also implemented in tensorFlow-privacy https://github.com/tensorflow/privacy
    
    clf: fitted model.
    X_test: test data.
    y_test: test data labels.
    """
    y_pred = clf.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    tpr =  tp / (tp + fn) #true positive rate or recall
    fpr =  fp / (fp + tn) #false positive rate
    return abs(tpr - fpr)
