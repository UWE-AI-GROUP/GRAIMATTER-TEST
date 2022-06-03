#!/usr/bin/python3

"""
Attribute inference attacks.
"""

from __future__ import annotations

import logging
import os
import pickle
from typing import Any, Final

import matplotlib.pyplot as plt
import multiprocess as mp
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from .data import Data, get_aia_data

logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.basicConfig()
logger = logging.getLogger("aia")
logger.setLevel(logging.WARNING)

N_CPU: int = mp.cpu_count()  # number of CPU cores to use
COLOR_A: str = "#86bf91"  # training set plot colour
COLOR_B: str = "steelblue"  # testing set plot colour


def get_inference_data(
    model: BaseEstimator, ds: Data, feature_id: int, memberset: bool
) -> tuple[np.ndarray, np.ndarray, float]:
    """Returns a dataset of each sample with the attributes to test."""
    attack_feature: dict = ds.features[feature_id]
    indices: list[int] = attack_feature["indices"]
    unique = np.unique(ds.x[:, feature_id])
    n_unique: int = len(unique)
    if attack_feature["encoding"] == "onehot":
        onehot_enc = OneHotEncoder()
        values = onehot_enc.fit_transform(unique.reshape(-1, 1)).toarray()
    else:
        values = unique
    samples: np.ndarray = ds.x_train  # samples after encoding (e.g. one-hot)
    samples_orig: np.ndarray = ds.Xt_member  # samples before encoding (e.g. str)
    if not memberset:
        samples = ds.x_test
        samples_orig = ds.Xt_nonmember
    n_samples, x_dim = np.shape(samples)
    x_values = np.zeros((n_samples, n_unique, x_dim), dtype=np.float64)
    y_values = model.predict(samples)
    # for each sample to perform inference on
    # add each possible missing feature value
    for i, x in enumerate(samples):
        for j, value in enumerate(values):
            x_values[i][j] = np.copy(x)
            x_values[i][j][indices] = value
    _, counts = np.unique(samples_orig[:, feature_id], return_counts=True)
    baseline = (np.max(counts) / n_samples) * 100
    logger.debug("x_values shape = %s", np.shape(x_values))
    logger.debug("y_values shape = %s", np.shape(y_values))
    return x_values, y_values, baseline


def unique_max(confidences: list[float], threshold: float) -> bool:
    """Returns whether there is a unique maximum confidence value above
    threshold."""
    if len(confidences) > 0:
        max_conf = np.max(confidences)
        if max_conf < threshold:
            return False
        unique, count = np.unique(confidences, return_counts=True)
        for (u, c) in zip(unique, count):
            if c == 1 and u == max_conf:
                return True
    return False


def infer(
    model: BaseEstimator,
    ds: Data,
    feature_id: int,
    threshold: float,
    memberset: bool,
) -> tuple[int, int, float, int, int]:
    """
    For each possible missing value, compute the confidence scores and
    label with the target model; if the label matches the known target model
    label for the original sample, and the highest confidence score is unique,
    infer that attribute if the confidence score is greater than a threshold.
    """
    correct: int = 0  # number of correct inferences made
    total: int = 0  # total number of inferences made
    x_values, y_values, baseline = get_inference_data(model, ds, feature_id, memberset)
    n_unique: int = len(x_values[1])
    samples = ds.x_train
    if not memberset:
        samples = ds.x_test
    n_samples: int = len(samples)

    for i, x in enumerate(x_values):  # each sample to perform inference on
        # get model confidence scores for all possible values for the sample
        confidence = model.predict_proba(x)
        # get known target model predicted label for the original sample
        label = y_values[i]
        conf = []  # confidences for each possible value with correct label
        attr = []  # features for each possible value with correct label
        # for each possible attribute value,
        # if the label matches the target model label
        # then store the confidence score and the tested feature vector
        for j in range(n_unique):
            this_label = np.argmax(confidence[j])
            scores = confidence[j][this_label]
            if this_label == label:
                conf.append(scores)
                attr.append(x[j])
        # is there is a unique maximum confidence score above threshold?
        if unique_max(conf, threshold):
            total += 1
            inf = attr[np.argmax(conf)]  # inferred feature vector
            if (inf == samples[i]).all():
                correct += 1
    return correct, total, baseline, n_unique, n_samples


def report_categorical(results: list[dict]) -> None:
    """Prints categorical results in a report format."""
    for feature in results:
        name = feature["name"]
        _, _, _, n_unique, _ = feature["train"]
        msg = f"Attacking categorical feature {name} with {n_unique} unique values:\n"
        for tranche in ("train", "test"):
            correct, total, baseline, _, n_samples = feature[tranche]
            if total > 0:
                msg += (
                    f"Correctly inferred {(correct / total) * 100:.2f}% "
                    f"of {(total / n_samples) * 100:.2f}% of the {tranche} set; "
                    f"baseline: {baseline:.2f}%\n"
                )
            else:
                msg += f"Unable to make any inferences of the {tranche} set\n"
        print(msg)


def report_continuous(results: list[dict]) -> None:
    """Prints continuous results in a report format."""
    for feature in results:
        print(
            f"{feature['name']}: "
            f"{feature['train']:.2f} train risk, "
            f"{feature['test']:.2f} test risk"
        )


def plot_continuous_risk(res: dict, savefile: str = "") -> None:
    """Generates bar chart showing continuous value risk scores."""
    results = res["continuous"]
    if len(results) < 1:
        return
    dataset_name = res["name"]
    x = np.arange(len(results))
    ya = []
    yb = []
    names = []
    for feature in results:
        names.append(feature["name"])
        ya.append(feature["train"] * 100)
        yb.append(feature["test"] * 100)
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=90)
    ax.set_ylim([0, 100])
    ax.bar(x + 0.2, ya, 0.4, align="center", color=COLOR_A, label="train set")
    ax.bar(x - 0.2, yb, 0.4, align="center", color=COLOR_B, label="test set")
    title = "Percentage of Set at Risk for Continuous Attributes"
    ax.set_title(f"{dataset_name}\n{title}")
    ax.tick_params(axis="x", labelsize=10)
    ax.tick_params(axis="y", labelsize=10)
    ax.grid(linestyle="dotted", linewidth=1)
    ax.legend(loc="best")
    plt.margins(y=0)
    plt.tight_layout()
    plt.show()
    if savefile != "":
        postfix = "_continuous_risk.png"
        fig.savefig(savefile + postfix, pad_inches=0, bbox_inches="tight")
        logger.debug("Saved continuous risk plot: %s", savefile)


def plot_categorical_risk(res: dict, savefile: str = "") -> None:
    """Generates bar chart showing categorical risk scores."""
    results: list[dict] = res["categorical"]
    if len(results) < 1:
        return
    dataset_name: str = res["name"]
    x: np.ndarray = np.arange(len(results))
    ya: list[float] = []
    yb: list[float] = []
    names: list[str] = []
    for feature in results:
        names.append(feature["name"])
        correct_a, total_a, baseline_a, _, _ = feature["train"]
        correct_b, total_b, baseline_b, _, _ = feature["test"]
        a = ((correct_a / total_a) * 100) - baseline_a if total_a > 0 else 0
        b = ((correct_b / total_b) * 100) - baseline_b if total_b > 0 else 0
        ya.append(a)
        yb.append(b)
    horizontal: bool = False
    if horizontal:
        fig, ax = plt.subplots(1, 1, figsize=(5, 8))
        ax.set_yticks(x)
        ax.set_yticklabels(names)
        ax.set_xlim([-100, 100])
        ax.barh(x + 0.2, ya, 0.4, align="center", color=COLOR_A, label="train set")
        ax.barh(x - 0.2, yb, 0.4, align="center", color=COLOR_B, label="test set")
    else:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=90)
        ax.set_ylim([-100, 100])
        ax.bar(x + 0.2, ya, 0.4, align="center", color=COLOR_A, label="train set")
        ax.bar(x - 0.2, yb, 0.4, align="center", color=COLOR_B, label="test set")
    title: str = "Improvement Over Most Common Value Estimate"
    ax.set_title(f"{dataset_name}\n{title}")
    ax.tick_params(axis="x", labelsize=10)
    ax.tick_params(axis="y", labelsize=10)
    ax.grid(linestyle="dotted", linewidth=1)
    ax.legend(loc="best")
    plt.margins(y=0)
    plt.tight_layout()
    plt.show()
    if savefile != "":
        postfix = "_categorical_risk.png"
        fig.savefig(savefile + postfix, pad_inches=0, bbox_inches="tight")
        logger.debug("Saved categorical risk plot: %s", savefile)


def plot_categorical_fraction(res: dict, savefile: str = "") -> None:
    """Generates bar chart showing fraction of dataset inferred."""
    results: list[dict] = res["categorical"]
    if len(results) < 1:
        return
    dataset_name: str = res["name"]
    x: np.ndarray = np.arange(len(results))
    ya: list[float] = []
    yb: list[float] = []
    names: list[str] = []
    for feature in results:
        names.append(feature["name"])
        _, total_a, _, _, n_samples_a = feature["train"]
        _, total_b, _, _, n_samples_b = feature["test"]
        a = ((total_a / n_samples_a) * 100) if n_samples_a > 0 else 0
        b = ((total_b / n_samples_b) * 100) if n_samples_b > 0 else 0
        ya.append(a)
        yb.append(b)
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=90)
    ax.set_ylim([0, 100])
    ax.bar(x + 0.2, ya, 0.4, align="center", color=COLOR_A, label="train set")
    ax.bar(x - 0.2, yb, 0.4, align="center", color=COLOR_B, label="test set")
    title: str = "Percentage of Set at Risk"
    ax.set_title(f"{dataset_name}\n{title}")
    ax.tick_params(axis="x", labelsize=10)
    ax.tick_params(axis="y", labelsize=10)
    ax.grid(linestyle="dotted", linewidth=1)
    ax.legend(loc="best")
    plt.margins(y=0)
    plt.tight_layout()
    plt.show()
    if savefile != "":
        postfix = "_categorical_fraction.png"
        fig.savefig(savefile + postfix, pad_inches=0, bbox_inches="tight")
        logger.debug("Saved categorical fraction plot: %s", savefile)


def plot_from_file(filename: str, savefile: str = "") -> None:
    """Loads a results save file and plots risk scores."""
    with open(filename + ".pickle", "rb") as handle:
        results = pickle.load(handle)
    plot_categorical_risk(results, savefile=savefile)
    plot_categorical_fraction(results, savefile=savefile)
    plot_continuous_risk(results, savefile=savefile)


def infer_categorical(
    model: BaseEstimator, ds: Data, feature_id: int, threshold: float
) -> dict:
    """Returns a the training and test set risks of a categorical feature."""
    result: dict = {
        "name": ds.features[feature_id]["name"],
        "train": infer(model, ds, feature_id, threshold, True),
        "test": infer(model, ds, feature_id, threshold, False),
    }
    return result


def is_categorical(ds: Data, feature_id: int) -> bool:
    """Returns whether a feature is categorical.
    For simplicity, assumes integer datatypes are categorical."""
    encoding: str = ds.features[feature_id]["encoding"]
    if encoding[:3] in ("str", "int") or encoding[:6] in ("onehot"):
        return True
    return False


def attack_brute_force(
    model: BaseEstimator, ds: Data, features: list[int]
) -> list[dict]:
    """
    Performs a brute force attribute inference attack by computing the target
    model confidence scores for every value in the list and making an inference
    if there is a unique highest confidence score.
    """
    attack_threshold: float = 0  # infer if unique highest conf exceeds
    ########
    # non-parallel single feature for testing
    # feature_id = features[0]
    # result = infer_categorical(model, ds, feature_id, attack_threshold)
    # print_result_categorical(result)
    # exit()
    ########
    args = [(model, ds, feature_id, attack_threshold) for feature_id in features]
    with mp.Pool(processes=N_CPU) as pool:
        results = pool.starmap(infer_categorical, args)
    return results


def fit(
    name: str,
    model: BaseEstimator,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> None:
    """Fits a model and displays the training and test scores."""
    model.fit(x_train, y_train)
    print(
        f"({model.score(x_train, y_train):.5f}, "
        f"{model.score(x_test, y_test):.5f})"
        f" {name} accuracy"
    )


def get_bounds_risk_for_sample(
    model: BaseEstimator,
    feat_id: int,
    feat_min: float,
    feat_max: float,
    sample: np.ndarray,
    c_min: float = 0,
    protection_limit: float = 0.1,
) -> bool:
    """Returns bool based on conditions surrounding upper and lower bounds of
    guesses that would lead to the same model confidence.

    model: trained target model.
    feat_id: index of missing feature.
    feat_min: minimum value of missing feature.
    feat_max: maximum value of missing feature.
    sample: original known feature vector.
    c_min: defines the confidence threshold below which we say we don't care.
    protection_limit: lower [upper] bound on estimated value must not be
    above[below] lower[upper] bounds e.g. 10% of value.
    """
    feat_n: int = 100  # number of attribute values to test per sample
    # attribute values to test - linearly sampled
    x_feat = np.linspace(feat_min, feat_max, feat_n, endpoint=True)
    # get known label
    label: int = int(model.predict(sample.reshape(1, -1))[0])
    # a matrix containing feature vector with linearly spaced target attribute
    x1 = np.repeat(sample.reshape(1, -1), feat_n, axis=0)
    x1[:, feat_id] = x_feat
    # get the target model confidences across the attribute range
    confidences = model.predict_proba(x1)
    scores = confidences[:, label]  # scores just for the model predicted label
    peak: float = np.max(scores)
    # find lowest and highest values with peak confidence
    lower_bound_index: int = 0
    while scores[lower_bound_index] < peak:
        lower_bound_index += 1
    upper_bound_index: int = feat_n - 1
    while scores[upper_bound_index] < peak:
        upper_bound_index -= 1
    # condition 1: confidence in prediction above some threshold
    # condition 2: confidence for true value == max_confidence
    # condition 3: lower bound above lower protection limit
    # condition 4: upper boiund of estiamte below upper protection limit
    actual_value = sample[feat_id]
    actual_probs = model.predict_proba(sample.reshape(1, -1))[0]
    lower_bound: float = x_feat[lower_bound_index]
    upper_bound: float = x_feat[upper_bound_index]
    if (
        peak > c_min
        and actual_probs[label] == peak
        and lower_bound >= protection_limit * actual_value
        and upper_bound <= (1 + protection_limit) * actual_value
    ):
        return True
    return False


def get_bounds_risk_for_feature(
    model: BaseEstimator, feature_id: int, samples: np.ndarray
) -> float:
    """Returns the average feature risk score over a set of samples."""
    feature_risk: int = 0
    n_samples: int = len(samples)
    feat_min: float = np.min(samples[:, feature_id])
    feat_max: float = np.max(samples[:, feature_id])
    for i in range(n_samples):
        sample = samples[i]
        risk = get_bounds_risk_for_sample(model, feature_id, feat_min, feat_max, sample)
        if risk:
            feature_risk += 1
    return feature_risk / n_samples


def get_bounds_risk(
    model: BaseEstimator,
    feature_name: str,
    feature_id: int,
    x_train: np.ndarray,
    x_test: np.ndarray,
) -> dict:
    """Returns a dictionary containing the training and test set risks of a
    continuous feature."""
    risk: dict = {
        "name": feature_name,
        "train": get_bounds_risk_for_feature(model, feature_id, x_train),
        "test": get_bounds_risk_for_feature(model, feature_id, x_test),
    }
    return risk


def get_bounds_risks(model: BaseEstimator, ds: Data, features: list[int]) -> list[dict]:
    """Computes the bounds risk for all specified features."""
    args = [
        (model, ds.features[feature_id]["name"], feature_id, ds.x_train, ds.x_test)
        for feature_id in features
    ]
    with mp.Pool(processes=N_CPU) as pool:
        results = pool.starmap(get_bounds_risk, args)
    return results


def get_bb_data(
    model: BaseEstimator, ds: Data, feature_id: int, seed: int | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Any]:
    """Returns data for fitting a model on n-1 features plus prediction."""
    # combine predictions with n-1 features for attack model input
    xa = np.copy(ds.x_train)
    predictions = model.predict_proba(xa)  # get target model's confidences
    attack_feature = ds.features[feature_id]
    xa = np.delete(xa, attack_feature["indices"], axis=1)  # drop attacked attr
    xa = np.concatenate((xa, predictions), axis=1)  # combine label predictions
    # get target vector of attacked feature
    ya = ds.Xt_member[:, feature_id]
    # encode target vector
    if is_categorical(ds, feature_id):
        encoder = LabelEncoder()
        ya = encoder.fit_transform(ya)
    else:
        encoder = StandardScaler()
        xa = encoder.fit_transform(xa)
        ya = encoder.fit_transform(ya.reshape(-1, 1)).ravel()
    # attack model train / test split
    xa_train, xa_test, ya_train, ya_test = train_test_split(
        xa, ya, test_size=0.2, shuffle=False, random_state=seed
    )
    return xa_train, ya_train, xa_test, ya_test, encoder


def attack_n_minus_1(model: BaseEstimator, ds: Data, seed: int | None) -> None:
    """Inference mapping n-1 features and labels to the missing feature."""
    attack_classifiers = [["RandomForestClassifier", RandomForestClassifier()]]
    attack_regressors = [["MLPRegressor", MLPRegressor(max_iter=1000)]]
    for i in range(ds.n_features):
        print(f"Attacking feature: {ds.features[i]['name']}")
        x_train, y_train, x_test, y_test, encoder = get_bb_data(model, ds, i, seed)
        if isinstance(encoder, LabelEncoder):
            for name, attack_model in attack_classifiers:
                fit(name, attack_model, x_train, y_train, x_test, y_test)
        else:
            for name, attack_model in attack_regressors:
                fit(name, attack_model, x_train, y_train, x_test, y_test)
        print("")


def attribute_inference(
    model: BaseEstimator,
    ds: Data,
    report: bool = False,
    savefile: str = "",
) -> dict:
    """
    Execute attribute inference attacks on a dataset given a trained model.
    """
    # brute force attack categorical attributes using dataset unique values
    logger.debug("Attacking dataset: %s", ds.name)
    logger.debug("Attacking categorical attributes...")
    feature_list: list[int] = []
    for feature in range(ds.n_features):
        if is_categorical(ds, feature):
            feature_list.append(feature)
    results_a: list[dict] = attack_brute_force(model, ds, feature_list)
    # compute risk scores for continuous attributes
    logger.debug("Attacking continuous attributes...")
    feature_list = []
    for feature in range(ds.n_features):
        if not is_categorical(ds, feature):
            feature_list.append(feature)
    results_b: list[dict] = get_bounds_risks(model, ds, feature_list)
    # display report output
    if report:
        report_categorical(results_a)
        report_continuous(results_b)
    # combine results into single object
    results: dict = {
        "name": ds.name,
        "categorical": results_a,
        "continuous": results_b,
    }
    # write to file
    if savefile != "":
        path = os.path.normpath(savefile + ".pickle")
        with open(path, "wb") as handle:
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
            logger.debug("Pickled output: %s.pickle", savefile)
    return results


if __name__ == "__main__":
    SEED: Final[int] = 1
    NAME: Final[str] = "in-hospital-mortality"
    FILENAME: Final[str] = "test"
    logger.setLevel(logging.DEBUG)

    # plot_from_file(FILENAME, savefile="hospital")
    # exit()

    # get dataset
    dataset = get_aia_data(NAME, random_state=SEED)
    print(dataset.data.describe())
    # train model
    target = RandomForestClassifier(bootstrap=False, random_state=SEED)
    target.fit(dataset.x_train, dataset.y_train)
    acc_train = target.score(dataset.x_train, dataset.y_train)
    acc_test = target.score(dataset.x_test, dataset.y_test)
    print(f"Base model train accuracy: {acc_train}")
    print(f"Base model test accuracy: {acc_test}")
    # perform attacks
    aia_results = attribute_inference(target, dataset, savefile=FILENAME)
    # plot results
    plot_categorical_risk(aia_results, savefile=FILENAME)
    plot_categorical_fraction(aia_results, savefile=FILENAME)
    plot_continuous_risk(aia_results, savefile=FILENAME)
