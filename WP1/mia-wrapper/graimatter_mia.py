import numpy as np

from tensorflow.keras import Model

# scikit-learn classifiers
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, DotProduct, Matern, RationalQuadratic, WhiteKernel
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier

from sklearn.base import ClassifierMixin
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from typing import Tuple, Union


Sklearn_classifier = Union[MLPClassifier,
                           KNeighborsClassifier,
                           SVC,
                           GaussianProcessClassifier,
                           DecisionTreeClassifier,
                           RandomForestClassifier,
                           AdaBoostClassifier,
                           XGBClassifier]
Classifier = Union[Sklearn_classifier, Model]
Labeled_Dataset = Tuple[np.ndarray, np.ndarray]


def generate_synthetic_data(n_samples: int,
                            n_classes: int,
                            n_features: int) -> Tuple[Labeled_Dataset, Labeled_Dataset, Labeled_Dataset]:
    """
    Generates synthetic data using the sklearn `make_classification` function.
    Three synthetic datasets are generated: a target dataset and a shadow dataset
    coming from the same distribution and an additional dataset from a different
    distribution.

    :param n_samples: Number of samples for each dataset.
    :param n_classes: Number of classes.
    :param n_features: number of informative features in the data.
    :return: A tuple with three datasets. Each of the datasets is a
        tuple with two numpy arrays for features and labels.
    """

    # Target distribution
    x, y = make_classification(
        n_samples=n_samples * 2,
        n_classes=n_classes,
        n_features=n_features,
        n_informative=n_features,
        n_redundant=0,
        n_repeated=0
    )

    # (xt, yt) is the target dataset, owned by the TRE and drawn from the (x,y) distribution
    # (xs, ys) is a shadow dataset drawn from the (x,y) distribution
    xt, xs, yt, ys = train_test_split(x, y, test_size=0.50, shuffle=False)

    # (xd, yd) is a shadow dataset, drawn from a different distribution (different seed)
    xd, yd = make_classification(
        n_samples=n_samples,
        n_classes=n_classes,
        n_features=300,
        n_informative=300,
        n_redundant=0,
        n_repeated=0
    )

    return (xt, yt), (xs, ys), (xd, yd)


def get_attack_data(model: Classifier, n_classes: int, train_data: np.ndarray,
                    test_data: np.ndarray) -> Labeled_Dataset:
    """
    Get classification probabilities from the target model or from
    a shadow model.

    :param model: A trained Scikit-learn or TensorFlow classifier.
    :param n_classes: The number of classes.
    :param train_data: Data used for training (member instances).
    :param test_data: Data used for testing (non-member instances).
    :return: Attack data with probabilities and membership labels.
    """
    assert (n_classes >= 2, "The number of classes must be at least 2.")

    if issubclass(type(model), ClassifierMixin):
        # Get prediction probabilities from sklearn model
        probabilities = np.vstack(
            (
                model.predict_proba(train_data),
                model.predict_proba(test_data)
            )
        )
    elif issubclass(type(model), Model):
        # Get prediction probabilities from tensorflow model
        probabilities = np.vstack(
            (
                model.predict(train_data),
                model.predict(test_data)
            )
        )
    else:
        raise Exception("Model must be a trained scikit-learn or TensorFlow classifier.")

    # Keep only the 3  highest probabilities in descending order.
    # Keep only 2 if binary classifier
    probabilities = np.sort(probabilities, axis=1)
    probabilities = np.flip(probabilities, axis=1)
    n_probs = 3 if n_classes > 2 else 2
    probabilities = probabilities[:, :n_probs]

    # Membership labels
    membership = np.vstack(
        (
            np.ones((train_data.shape[0], 1), np.uint8),
            np.zeros((test_data.shape[0], 1), np.uint8)
        )
    ).flatten()

    return probabilities, membership


def train_attack_model(probabilities: np.ndarray, labels: np.ndarray) -> Sklearn_classifier:
    """
    Train an attack model from the classification probabilities and membership information.

    :param probabilities: Classification probabilities obtained from target or shadow model.
    :param labels: Membership information (members/nonmembers).
    :return: The attack model (a scikit-learn MLPClassifier).
    """

    mlp_clf = MLPClassifier(max_iter=500)

    param_grid = {
        'hidden_layer_sizes': [(64,), (32, 32), (32, 32, 32)],
        'solver': ['sgd', 'adam'],
        'alpha': [0.0001, 0.001, 0.01],
    }
    n_jobs = -1

    attack_model = GridSearchCV(mlp_clf, param_grid=param_grid, cv=3, n_jobs=n_jobs, refit=True, verbose=0)
    attack_model.fit(probabilities, labels)

    return attack_model.best_estimator_


def evaluate_attack(target_model: Classifier,
                    attack_model: Sklearn_classifier,
                    members: np.ndarray,
                    nonmembers: np.ndarray,
                    n_classes: int) -> Tuple[float, float, float, float]:
    """

    :param target_model: Trained Scikit-learn or Tensorflow model to attack.
    :param attack_model: Attack model, distinguishes from members and nonmembers.
    :param members: Data used to train the target model (members).
    :param nonmembers: Evaluation data (nonmembers).
    :param n_classes: Number of classes.
    :return: Attack performance results.
    """

    probabilities, true_membership = get_attack_data(target_model, n_classes, members, nonmembers)

    predicted_membership = attack_model.predict(probabilities)

    print(classification_report(true_membership, predicted_membership, target_names=['nonmember', 'member'], digits=4))
    cm = confusion_matrix(true_membership, predicted_membership)
    tn = cm[0, 0]
    tp = cm[1, 1]
    fn = cm[0, 1]
    fp = cm[1, 0]

    return tn, tp, fn, fp


def evaluate_model_privacy(target_model: Classifier, x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float, float]:
    pass

def _train_MLPClassifier(x: np.ndarray, y: np.ndarray) -> Classifier:
    """
    Trains a scikit-learn multilayer perceptron classifier using GridSearch.

    :param x: Features
    :param y: Labels
    :return: A trained MLPClassifier
    """
    base_clf = MLPClassifier(max_iter=500)

    param_grid = {
        'hidden_layer_sizes': [(64,), (32, 32), (32, 32, 32)],
        'solver': ['sgd', 'adam'],
        'alpha': [0.0001, 0.001, 0.01],
    }
    n_jobs = -1

    clf = GridSearchCV(base_clf, param_grid=param_grid, cv=3, n_jobs=n_jobs, refit=True, verbose=0)
    clf.fit(x, y)

    return clf.best_estimator_

def _train_KNeighborsClassifier(x: np.ndarray, y: np.ndarray) -> Classifier:
    """
    Trains a scikit-learn k-Nearest Neighbours classifier using GridSearch.

    :param x: Features
    :param y: Labels
    :return: A trained KNeighborsClassifier
    """
    base_clf = KNeighborsClassifier()

    param_grid = {
        'n_neighbors': [3, 5, 11, 19],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }

    n_jobs = -1

    clf = GridSearchCV(base_clf, param_grid=param_grid, cv=3, n_jobs=n_jobs, refit=True, verbose=0)
    clf.fit(x, y)

    return clf.best_estimator_

def _train_SVC(x: np.ndarray, y: np.ndarray) -> Classifier:
    """
    Trains a scikit-learn Support Vector Machine classifier using GridSearch.

    :param x: Features
    :param y: Labels
    :return: A trained SVC
    """
    base_clf = SVC(probability=True)

    param_grid = {
        'C': [1, 10, 100, 1000],
        'gamma': [1, 0.1, 0.001, 0.0001],
        'kernel': ['linear', 'rbf']
    }

    n_jobs = -1

    clf = GridSearchCV(base_clf, param_grid=param_grid, cv=3, n_jobs=n_jobs, refit=True, verbose=0)
    clf.fit(x, y)

    return clf.best_estimator_

def _train_GaussianProcessClassifier(x: np.ndarray, y: np.ndarray) -> Classifier:
    """
    Trains a scikit-learn Gaussian Process classifier using GridSearch.

    :param x: Features
    :param y: Labels
    :return: A trained GaussianProcessClassifier
    """

    base_clf = GaussianProcessClassifier()

    param_grid = {
        'kernel': [1*RBF(), 1*DotProduct(), 1*Matern(),  1*RationalQuadratic(), 1*WhiteKernel()]
    }
    n_jobs = -1

    clf = GridSearchCV(base_clf, param_grid=param_grid, cv=3, n_jobs=n_jobs, refit=True, verbose=0)
    clf.fit(x, y)

    return clf.best_estimator_

def _train_DecisionTreeClassifier(x: np.ndarray, y: np.ndarray) -> Classifier:
    """
    Trains a scikit-learn Decision Tree classifier using GridSearch.

    :param x: Features
    :param y: Labels
    :return: A trained DecisionTreeClassifier
    """

    base_clf = DecisionTreeClassifier()

    param_grid = {
        'max_features': ['auto', 'sqrt', 'log2'],
        'ccp_alpha': [0.1, .01, .001],
        'max_depth' : [5, 6, 7, 8, 9],
        'criterion' :['gini', 'entropy']
    }
    n_jobs = -1

    clf = GridSearchCV(base_clf, param_grid=param_grid, cv=3, n_jobs=n_jobs, refit=True, verbose=0)
    clf.fit(x, y)

    return clf.best_estimator_

def _train_RandomForestClassifier(x: np.ndarray, y: np.ndarray) -> Classifier:
    """
    Trains a scikit-learn Random Forest classifier using GridSearch.

    :param x: Features
    :param y: Labels
    :return: A trained RandomForestClassifier
    """

    base_clf = RandomForestClassifier()

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt', 'log2'],
        'max_depth': [4, 5, 6, 7, 8],
        'criterion': ['gini', 'entropy']
    }
    n_jobs = -1

    clf = GridSearchCV(base_clf, param_grid=param_grid, cv=3, n_jobs=n_jobs, refit=True, verbose=0)
    clf.fit(x, y)

    return clf.best_estimator_

def _train_AdaBoostClassifier(x: np.ndarray, y: np.ndarray) -> Classifier:
    """
    Trains a scikit-learn Ada Boost classifier using GridSearch.

    :param x: Features
    :param y: Labels
    :return: A trained AdaBoostClassifier
    """

    base_clf = AdaBoostClassifier()

    param_grid = {
     'n_estimators': np.arange(10, 300, 10),
     'learning_rate': [0.01, 0.05, 0.1, 1]
    }
    n_jobs = -1

    clf = GridSearchCV(base_clf, param_grid=param_grid, cv=3, n_jobs=n_jobs, refit=True, verbose=0)
    clf.fit(x, y)

    return clf.best_estimator_

def _train_XGBClassifier(x: np.ndarray, y: np.ndarray) -> Classifier:
    """
    Trains a scikit-learn XGBoost classifier using GridSearch.

    :param x: Features
    :param y: Labels
    :return: A trained XGBClassifier
    """

    base_clf = XGBClassifier()

    param_grid = {
        'max_depth': range (2, 10, 1),
        'n_estimators': range(60, 220, 40),
        'learning_rate': [0.1, 0.01, 0.05]
    }
    n_jobs = -1

    clf = GridSearchCV(base_clf, param_grid=param_grid, cv=3, n_jobs=n_jobs, scoring='roc_auc', refit=True, verbose=0)
    clf.fit(x, y)

    return clf.best_estimator_

def _train_TFNeuralNetwork(architecture: Model, x: np.ndarray, y: np.ndarray) -> Classifier:
    pass