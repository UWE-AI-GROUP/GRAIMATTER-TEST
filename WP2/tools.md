# Notes on open source privacy assessment tools

## Assessment Criteria

* "[Libraries.io](https://libraries.io) indexes data from 5,086,238 packages from 32 package managers. We monitor package releases, analyse each project's code, community, distribution and documentation, and we map the relationships between packages when they're declared as a dependency."
* It provides a searchable index of packages via keyword - perhaps we can specify some keywords and then systematically go through the packages to check for relevant ones - the keywords used will help other researchers know what we searched for? List so far ["trusted-ai", "trustworthy-ai", "poisoning", "privacy-protection", "privacy-enhancing-technologies"]. Same approach applied to github search?
* It generates a "source rank" for many projects, which is based on a set of criteria that include the project documentation, licenses, recency and number of versions available, whether [semantic version numbering](https://semver.org) is used, popularity measures such as github stars, the number of contributors, and how many packages or repositories are dependent on the project.

Following this general criteria, below are notes on some tools:

---------------------

## [TensorFlow Privacy](https://github.com/tensorflow/privacy)

* License: Apache-2.0
* Source: available on github
* Distribution: available in PyPi
* Popularity: 1.5k stars; 42 contributors
* Versioning: 17 releases using semantic version numbers (first release Aug 23, 2019; current version 0.7.3 released Sep 1, 2021)
* [Libraries.io](https://libraries.io/pypi/tensorflow-privacy) SourceRank: 14
* Dependants: 5 github projects depend on the package, including: [ydata-synthetic](https://github.com/ydataai/ydata-synthetic) and [gretel-synthetics](https://github.com/gretelai/gretel-synthetics) and [TensorFlow Federated](https://www.tensorflow.org/federated)
* Description:  Provides [differential privacy](https://github.com/tensorflow/privacy/tree/master/tutorials) (requires setting parameters that control the way gradients are created, clipped, and noised) and empirical [privacy tests](https://github.com/tensorflow/privacy/tree/master/tensorflow_privacy/privacy/privacy_tests) for measuring potential memorisation. The tests include [secret sharer attacks](https://arxiv.org/abs/1802.08232) and [membership inference attacks](https://github.com/tensorflow/privacy/tree/master/tensorflow_privacy/privacy/privacy_tests/membership_inference_attack).
  - The membership inference attack API supports a limited number of classifier attacks, including `AttackType.LOGISTIC_REGRESSION`, `AttackType.MULTI_LAYERED_PERCEPTRON`, `AttackType.RANDOM_FOREST`, `AttackType.K_NEAREST_NEIGHBORS`, and threshold attacks `AttackType.THRESHOLD_ATTACK`, and `AttackType.THRESHOLD_ENTROPY_ATTACK`.
  - AUC and [advantage](https://arxiv.org/abs/1709.01604) metrics are provided for measuring attack success.
  - In threshold attacks, the number of training and testing samples that have membership score greater than a given threshold value are counted. The threshold value is usually set between 0.5 (random chance of member or non-member), and 1 (100% certainty of membership.) Precision and recall values are computed. Following [Song and Mittal, 2020](https://arxiv.org/abs/2003.10595), based on these values, an ROC curve can be produced representing how accurate the attacker can predict whether or not a data point was used in the training data.
  - For trained attacks, the attack flow involves training a shadow model.

---------------------

## [Adversarial Robustness Toolbox](https://github.com/Trusted-AI/adversarial-robustness-toolbox)

* License: MIT
* Source: available on github
* Distribution: available in PyPi
* Popularity: 2.7k stars; 75 contributors
* Versioning: 27 releases using semantic version numbers (first release Apr 25, 2018; current version 1.9.1 released Jan 7, 2022)
* [Libraries.io](https://libraries.io/pypi/adversarial-robustness-toolbox) SourceRank: 14
* Dependants: 6 github projects depend on the package, including: [IBM AI Fairness 360](https://github.com/Trusted-AI/AIF360) and [ML-PePR](https://github.com/hallojs/ml-pepr) and [TrojAI](https://pypi.org/project/troj/1.0.0/)
* Description: Provides a range of privacy attacks, including inference attacks and [model extraction](https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/notebooks/model-stealing-demo.ipynb). "ART supports all popular machine learning frameworks (TensorFlow, Keras, PyTorch, MXNet, scikit-learn, XGBoost, LightGBM, CatBoost, GPy, etc.), all data types (images, tables, audio, video, etc.) and machine learning tasks (classification, object detection, speech recognition, generation, certification, etc.)."
  - Provides three types of inference attacks:
    + [membership inference](https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/notebooks/attack_membership_inference.ipynb) attempts to determine if the information of a certain record, e.g. of a person, has been part of the training data of a trained ML model. The [API](https://adversarial-robustness-toolbox.readthedocs.io/en/latest/modules/attacks/inference/membership_inference.html) supports `MembershipInferenceBlackBox`, `MembershipInferenceBlackBoxRuleBased`, `LabelOnlyDecisionBoundary`, and `LabelOnlyGapAttack`.
    + [attribute inference](https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/notebooks/attack_attribute_inference.ipynb) aims at inferring the actual feature values of a record known to exist in the training data only by accessing the trained model and knowing few of the other features of the record. For example, a ML model trained on demographic data attacked with attribute inference could leak information about a person's exact age or salary. The [API](https://adversarial-robustness-toolbox.readthedocs.io/en/latest/modules/attacks/inference/attribute_inference.html) supports `AttributeInferenceBaseline`, `AttributeInferenceBlackBox`, `AttributeInferenceMembership`, `AttributeInferenceWhiteBoxLifestyleDecisionTree`, and `AttributeInferenceWhiteBoxDecisionTree`.
    + [model inversion](https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/notebooks/model_inversion_attacks_mnist.ipynb) aims to reconstruct representative averages of features of the training data by inverting a trained ML model. The [API](https://adversarial-robustness-toolbox.readthedocs.io/en/latest/modules/attacks/inference/model_inversion.html) supports `MIFace`, which provides an implementation of the MIFace algorithm from [Fredrikson et al. (2015)](https://doi.org/10.1145/2810103.2813677).

---------------------

## [Fawkes](https://github.com/Shawn-Shan/fawkes)

* License: BSD-3
* Source: available on github
* Distribution: available in PyPi
* Popularity: 4.4k stars; 3 contributors
* Versioning: single release (version 0.3 on 30 Jul, 2020)
* [Libraries.io](https://libraries.io/pypi/fawkes) SourceRank: 13
* Description: "we propose Fawkes, a system that helps individuals to inoculate their images against unauthorized facial recognition models at any time without significantly distorting their own photos, or wearing conspicuous patches. Fawkes achieves this by helping users adding imperceptible pixel-level changes (“cloaks”) to their own photos. For example, a user who wants to share content (e.g. photos) on social media or the public web can add small, imperceptible alterations to their photos before uploading them. If collected by a third-party “tracker” and used to train a facial recognition model to recognize the user, these “cloaked” images would produce functional models that consistently misidentify them."

---------------------

## [Diffprivlib](https://github.com/IBM/differential-privacy-library)

* License: MIT
* Source: available on github
* Distribution: available in PyPi
* Popularity: 555 stars; 10 contributors
* Versioning: 6 releases using semantic version numbers (first release Jun 19, 2019; current version 0.5.0 released Oct 1, 2021)
* [Libraries.io](https://libraries.io/pypi/diffprivlib) SourceRank: 12
* Dependants: 3 projects depend on the package: [IBM Watson](http://ibm-wml-api-pyclient.mybluemix.net/) and [synthetic-data-generation](https://github.com/daanknoors/synthetic_data_generation) and [spark-privacy-preserver](https://github.com/ThaminduR/spark-privacy-preserver)
* Description: Provides a set of wrappers implementing differential privacy for a small number of scikit-learn style models including, Gaussian Naive Bayes, Logistic Regression, Random Forest, Linear Regression, PCA, and K-Means clustering.

---------------------

## [IBM AI Privacy Toolkit](https://github.com/IBM/ai-privacy-toolkit)

* License: MIT
* Source: available on github
* Distribution: available in PyPi
* Popularity: 14 stars; 3 contributors
* Versioning: no tags have been generated to assign version numbers (and there is no version tagged as a release)
* [Libraries.io](https://libraries.io/pypi/ai-privacy-toolkit) SourceRank: 7

---------------------

## [ML-PePR](https://github.com/hallojs/ml-pepr)

* License: GPLv3
* Source: available on github
* Distribution: available in PyPi
* Popularity: 0 stars; 3 contributors
* Versioning: uses semantic version numbering with 6 releases (0.1b6 latest version on 17 Jul, 2021)
* [Libraries.io](https://libraries.io/pypi/mlpepr) SourceRank: 6

---------------------

## [ML Privacy Meter](https://github.com/privacytrustlab/ml_privacy_meter)

* License: MIT
* Source: available on github
* Distribution: no distributions available
* Popularity: 241 stars; 6 contributors
* Versioning: no tags have been generated to assign version numbers (and there is no version tagged as a release);
* Notes: The tool expects a Keras/TensorFlow model; the dataset must be in a specific format; while the documentation states that "the API is built on top of TensorFlow 2.1 with Python 3.6", most of the tutorials and utility scripts are in antiquated Python 2 (e.g., see [here](https://github.com/privacytrustlab/ml_privacy_meter/tree/master/datasets) "so you need to have Python 2 and numpy with Python 2 support"); the tool requires the setting of a number of model/dataset specific parameters including an optimiser, a list of layers to exploit, the learning rate, number of training epochs, which means that it will likely require both (a) human involvement to define the set of parameters to explore and (b) multiple runs of the tool for tuning;

---------------------

## [ML-Doctor](https://github.com/liuyugeng/ML-Doctor)

* License: Apache-2.0
* Source: available on github
* Distribution: no distributions available
* Popularity: 10 stars; 1 contributor
* Versioning: no tags have been generated to assign version numbers (and there is no version tagged as a release)

---------------------
