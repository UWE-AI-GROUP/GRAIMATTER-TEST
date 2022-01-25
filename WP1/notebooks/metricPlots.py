import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import auc, RocCurveDisplay
import numpy as np


def plot_confusion_matrix(name, confusion_matrix, classes,
                          normalize=False,
                          cmap=plt.cm.Blues,
                          save=None):
    """
    This function plots a confusion matrix for predictions of a given model.
    Name: is the name of the model.
    confusion_matrix: is the confusion matrix, e.g. output from confusion_matrix(y_test, y_pred).
    classes: is the number of classes, e.g. output of range(2).
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
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

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
    else:
        plt.show()


def plotROC_classifier(clf,
                       X_tmp_test, y_tmp_test,
                       title="ROC curve", save=None):
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
    else:
        fig.show()



def plot_prob_test_train(pred_test, pred_train, title='Membership probalibilty', save=None):
    """
    This function plots and histogram of the probability associated
    with MIA (memebership inference attack).

    pred_test: predicted probability of the test data
    (from the MIA split, not from building the model).
    pred_train: predicted probability of the train data
    (from the MIA split, not from building the model).
    title: Set a title for the figure, e.g. specify model name/parameters
    and MIA etc. Default "Membership probalibilty".
    save: it save figure to the given path/figname. Default don't save.
    """
    #fig, ax = plt.subplots()
    plt.hist(np.array(pred_train).flatten(),  alpha=0.5, bins=20, label='Training Data (Members)',
                histtype='bar', range=(0, 1))
    plt.hist(np.array(pred_test).flatten(),  alpha=0.5, bins=20, label='Test Data (Non-members)',
                histtype='bar', range=(0, 1))
    plt.legend(loc='center left', bbox_to_anchor=(0.5, -0.25))
    plt.xlabel('Membership Probability')
    plt.ylabel('Fraction')
    plt.title(title)

    if save:
        plt.savefig(save, bbox_inches='tight')
    else:
        plt.show()
