import os
import matplotlib.pyplot as plt
from sklearn.metrics import auc, RocCurveDisplay, DetCurveDisplay, det_curve, confusion_matrix#,  classification_report
import scikitplot as skplt
from typing import Any, Iterable, Optional

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

    plt.savefig(os.path.join(path, "Confusion_matrix_"+name+".png"), bbox_inches='tight')
    plt.show()

    
    
def plot_roc_curve(clf:Any,
                   X: Iterable[float], 
                   y: Iterable[float],
                   title: Optional[str] = "ROC curve",
                   path: Optional[str] = os.getcwd()):
    """
    This function calculates and plots a ROC AUC curve of a given model.

    clf: is the fitted classifier.
    X: test data.
    y: labels of test data.
    title: Set a title for the figure, e.g. specify model name/parameters etc. Default "ROC curve".
    path: it save figure to the given path/figname. Default don't save.
    """
    n_classes = y.shape[1]
    
    y_pred = model.predict(X)

    # roc curve for classes
    fpr = {}
    tpr = {}
    thresh ={}

    for i in range(n_class):    
        fpr[i], tpr[i], thresh[i] = roc_curve(y[:,i], y_pred[:,i], pos_label=1)
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], linestyle='--', label="Class %f ROC curve (area = %0.2f)" % roc_auc[i])

    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.title(title)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive rate')
    plt.legend(loc='best')
    plt.show()
    plt.savefig(path,dpi=300)
    
    
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

    plt.savefig(os.path.join(path, f'{title}_class_{plot_class}.png'), bbox_inches='tight')

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
    plt.savefig(os.path.join(path, title+"_"+model_name+".png"), bbox_inches='tight')
    plt.show()

    

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
    plt.savefig(os.path.join(path, title+".png"), bbox_inches='tight')
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
    plt.savefig(os.path.join(path, "calibration_curves.png"), bbox_inches='tight')
    plt.show()
