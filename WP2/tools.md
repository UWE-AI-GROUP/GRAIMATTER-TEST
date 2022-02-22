# Notes on open source privacy assessment tools

## Assessment Criteria

* Project license
* Project documentation and tutorials
* Project based on peer-reviewed publication(s)
* Project version numbered and has a version tagged as a release
* Project recency and update frequency
* Project popularity (e.g., GitHub stars and contributors)
* Project quality assurance (e.g., use of unit tests, static analysis, and continuous integration tools)
* Project distribution: location and ease of installation with current platforms/software
* Project ease of use: including output analysis metrics and/or visualisations
* Number of and quality of other projects that use/depend on the project

---------------------

## [TensorFlow Privacy](https://github.com/tensorflow/privacy)

* Category: Attacks (black-box) and Defences
* License: Apache-2.0
* Source: available on github
* Distribution: available in PyPi
* Popularity: 1.5k stars; 42 contributors
* Versioning: 17 releases using semantic version numbers (first release Aug 23, 2019; current version 0.7.3 released Sep 1, 2021)
* [Libraries.io](https://libraries.io/pypi/tensorflow-privacy) SourceRank: 14
* Dependants: 5 github projects depend on the package, including: [ydata-synthetic](https://github.com/ydataai/ydata-synthetic) and [gretel-synthetics](https://github.com/gretelai/gretel-synthetics) and [TensorFlow Federated](https://www.tensorflow.org/federated)
* Description:  Provides [differential privacy](https://github.com/tensorflow/privacy/tree/master/tutorials) (requires setting parameters that control the way gradients are created, clipped, and noised) and empirical [privacy tests](https://github.com/tensorflow/privacy/tree/master/tensorflow_privacy/privacy/privacy_tests) for measuring potential memorisation. The tests include [secret sharer attacks](https://arxiv.org/abs/1802.08232) and [membership inference attacks](https://github.com/tensorflow/privacy/tree/master/tensorflow_privacy/privacy/privacy_tests/membership_inference_attack).
  - The membership inference attack API supports a limited number of black-box classifier attacks, including `AttackType.LOGISTIC_REGRESSION`, `AttackType.MULTI_LAYERED_PERCEPTRON`, `AttackType.RANDOM_FOREST`, `AttackType.K_NEAREST_NEIGHBORS`, and threshold attacks `AttackType.THRESHOLD_ATTACK`, and `AttackType.THRESHOLD_ENTROPY_ATTACK`.
  - AUC and [advantage](https://arxiv.org/abs/1709.01604) (maximum false positive rate minus true positive rate) metrics are provided for measuring attack success.
  - In threshold attacks, the number of training and testing samples that have membership score greater than a given threshold value are counted. The threshold value is usually set between 0.5 (random chance of member or non-member), and 1 (100% certainty of membership.) Precision and recall values are computed. Following [Song and Mittal, 2020](https://arxiv.org/abs/2003.10595), based on these values, an [ROC curve](https://en.wikipedia.org/wiki/Receiver_operating_characteristic) (false positive vs. true positive rate) can be produced representing how accurate the attacker can predict whether or not a data point was used in the training data.
  - For trained attacks, the attack flow involves training a shadow model.

---------------------

## [Adversarial Robustness Toolbox](https://github.com/Trusted-AI/adversarial-robustness-toolbox)

* Category: Attacks (black-box and white-box) and Defences
* License: MIT
* Source: available on github
* Distribution: available in PyPi
* Popularity: 2.7k stars; 75 contributors
* Versioning: 27 releases using semantic version numbers (first release Apr 25, 2018; current version 1.9.1 released Jan 7, 2022)
* [Libraries.io](https://libraries.io/pypi/adversarial-robustness-toolbox) SourceRank: 14
* Dependants: 6 github projects depend on the package, including: [IBM AI Fairness 360](https://github.com/Trusted-AI/AIF360) and [ML-PePR](https://github.com/hallojs/ml-pepr) and [TrojAI](https://pypi.org/project/troj/1.0.0/)
* Description: Provides a range of privacy attacks, including inference attacks and [model extraction](https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/notebooks/model-stealing-demo.ipynb). "ART supports all popular machine learning frameworks (TensorFlow, Keras, PyTorch, MXNet, scikit-learn, XGBoost, LightGBM, CatBoost, GPy, etc.), all data types (images, tables, audio, video, etc.) and machine learning tasks (classification, object detection, speech recognition, generation, certification, etc.)." Defences are mostly focused on adversarial examples (where inputs are crafted to cause the network to misclassify), however it also includes defences against model stealing (e.g., using deceptive perturbations), and defences against [generative adversarial network](https://en.wikipedia.org/wiki/Generative_adversarial_network) attacks where the latent representation of an input image is inferred and the original input reconstructed (e.g., by improving classifier robustness to perturbation). Provides three types of inference attacks.
  - [membership inference](https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/notebooks/attack_membership_inference.ipynb) attempts to determine if the information of a certain record, e.g. of a person, has been part of the training data of a trained ML model. The [API](https://adversarial-robustness-toolbox.readthedocs.io/en/latest/modules/attacks/inference/membership_inference.html) supports a small number of black-box attacks:
    + `MembershipInferenceBlackBoxRuleBased` uses the simple rule to determine membership in the training data: if the model's prediction for a sample is correct, then it is a member. Otherwise, it is not a member.
    + `MembershipInferenceBlackBox` trains an additional classifier (the attack model) to predict the membership status of a sample. It can use as input to the learning process probabilities/logits or losses, depending on the type of model and provided configuration.
    + `LabelOnlyDecisionBoundary` based on [Li and Zhang (2021)](https://arxiv.org/abs/2007.15528) uses only the model predicted labels (no confidence scores) - the key intuition in the decision boundary attack is that it is harder to perturb member data samples to different classes than non-member data samples. The adversary queries the target model on candidate data samples, and perturbs them to change the model's predicted labels. The adversary can then exploit the magnitude of the perturbation to differentiate member and non-member samples.
    + `LabelOnlyGapAttack` alias of `MembershipInferenceBlackBoxRuleBased` - used in [Choquette-Choo et al. (2021)](https://arxiv.org/abs/2007.14321): predicts any misclassified data point as a non-member of the training set.
  - [attribute inference](https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/notebooks/attack_attribute_inference.ipynb) aims at inferring the actual feature values of a record known to exist in the training data only by accessing the trained model and knowing few of the other features of the record. For example, a ML model trained on demographic data attacked with attribute inference could leak information about a person's exact age or salary. The [API](https://adversarial-robustness-toolbox.readthedocs.io/en/latest/modules/attacks/inference/attribute_inference.html) supports `AttributeInferenceBaseline`, `AttributeInferenceBlackBox`, `AttributeInferenceMembership`, `AttributeInferenceWhiteBoxLifestyleDecisionTree`, and `AttributeInferenceWhiteBoxDecisionTree`. These are black-box attacks and decision tree white-box attacks based on [Fredrikson et al. (2015)](https://doi.org/10.1145/2810103.2813677).
  - [model inversion](https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/notebooks/model_inversion_attacks_mnist.ipynb) aims to reconstruct representative averages of features of the training data by inverting a trained ML model. The [API](https://adversarial-robustness-toolbox.readthedocs.io/en/latest/modules/attacks/inference/model_inversion.html) supports `MIFace`, which provides an implementation of the MIFace algorithm, also from [Fredrikson et al. (2015)](https://doi.org/10.1145/2810103.2813677).
  - [database reconstruction](https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/notebooks/attack_database_reconstruction.ipynb) attack assumes an adversary has in their possession a model trained on a dataset, and all but one row of that training dataset. The attack attempts to reconstruct the missing row.

---------------------

## [Fawkes](https://github.com/Shawn-Shan/fawkes)

* Category: Defences
* License: BSD-3
* Source: available on github
* Distribution: available in PyPi
* Popularity: 4.4k stars; 3 contributors
* Versioning: single release (version 0.3 on 30 Jul, 2020)
* [Libraries.io](https://libraries.io/pypi/fawkes) SourceRank: 13
* Description: "we propose Fawkes, a system that helps individuals to inoculate their images against unauthorized facial recognition models at any time without significantly distorting their own photos, or wearing conspicuous patches. Fawkes achieves this by helping users adding imperceptible pixel-level changes (“cloaks”) to their own photos. For example, a user who wants to share content (e.g. photos) on social media or the public web can add small, imperceptible alterations to their photos before uploading them. If collected by a third-party “tracker” and used to train a facial recognition model to recognize the user, these “cloaked” images would produce functional models that consistently misidentify them."

---------------------

## [Diffprivlib](https://github.com/IBM/differential-privacy-library)

* Category: Defences
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

* Category: Defences
* License: MIT
* Source: available on github
* Distribution: available in PyPi
* Popularity: 14 stars; 3 contributors
* Versioning: no tags have been generated to assign version numbers (and there is no version tagged as a release)
* [Libraries.io](https://libraries.io/pypi/ai-privacy-toolkit) SourceRank: 7
* Description: Provides an [anonymisation module](https://github.com/IBM/ai-privacy-toolkit/blob/main/notebooks/attribute_inference_anonymization_nursery.ipynb) and [minimisation module](https://github.com/IBM/ai-privacy-toolkit/blob/main/notebooks/minimization_adult.ipynb). The anonymisation module contains methods for anonymising training data, so that when a model is retrained on the anonymised data, the model itself will also be considered anonymous. The minimisation module contains methods to help adhere to the data minimisation principle in GDPR for ML models. It enables a reduction in the amount of personal data needed to perform predictions with a machine learning model, while still enabling the model to make accurate predictions. This is done by removing or generalising some of the input features.

---------------------

## [ML-PePR](https://github.com/hallojs/ml-pepr)

* Category: Attacks (black-box)
* License: GPLv3
* Source: available on github
* Distribution: available in PyPi
* Popularity: 0 stars; 3 contributors
* Versioning: uses semantic version numbering with 6 releases (0.1b6 latest version on 17 Jul, 2021)
* [Libraries.io](https://libraries.io/pypi/mlpepr) SourceRank: 6
* Description: Provides a [membership inference attack](https://colab.research.google.com/github/hallojs/ml-pepr/blob/master/notebooks/mia_tutorial.ipynb) based on [Shokri et al., (2017)](https://arxiv.org/abs/1610.05820) and a [direct generalised membership inference attack](https://colab.research.google.com/github/hallojs/ml-pepr/blob/master/notebooks/direct_gmia_tutorial.ipynb) based on [Long et al., (2018)](https://arxiv.org/abs/1802.04889).

---------------------

## [ML Privacy Meter](https://github.com/privacytrustlab/ml_privacy_meter)

* Category: Attacks (black-box and white-box)
* License: MIT
* Source: available on github
* Distribution: no distributions available
* Popularity: 241 stars; 6 contributors
* Versioning: no tags have been generated to assign version numbers (and there is no version tagged as a release)
* Description: Provides membership inference attacks of neural networks (fully-connected and convolutional) for both black-box and white-box scenarios based on [Nasr et al. (2019)](https://arxiv.org/abs/1812.00910). The Nasr et al. (2019) paper details the white-box attack as feeding the model parameters (gradients, activations, losses, and labels) from each specified layer as inputs to an autoencoder which compresses the model features before running a clustering algorithm to separate member and non-member samples. When comparing the differences between the black-box and white-box attack, the accuracy increases by less than 1% on one dataset to about 6.5% on another.
  - In addition to aggregated results, the tool reports the membership probability and accuracy of the attack per record.
  - The tool expects a Keras/TensorFlow model and the dataset must be in a specific format.
  - The tool requires the setting of a number of model/dataset specific parameters, which include (depending on white-box or black-box) an optimiser, a list of layers to exploit, the learning rate, and number of training epochs.
  - For black-box attacks, the output dimension of Model A and B must be identical (requires a one-hot encoding of labels), however the rest of the architecture can differ. For white-box attacks, the architectures must be identical.
  - Report generates a set of visualisations, including histograms for privacy risk, ROC curves for the membership probabilities, gradient norm distributions for member and non-member data, and label-wise privacy risk plots.
* Notes: the `requirements.txt` and tutorials are out of date. To use this tool, clone the repository and change to the root `ml_privacy_meter` directory; place source/notebooks here and execute; install any missing packages with pip; generated plots are located in `logs/plots`.

---------------------

## [ML-Doctor](https://github.com/liuyugeng/ML-Doctor)

* Category: Attacks (black-box and white-box)
* License: Apache-2.0
* Source: available on github
* Distribution: no distributions available
* Popularity: 11 stars; 1 contributor
* Versioning: no tags have been generated to assign version numbers (and there is no version tagged as a release)
* Description: Provides 10 black-box and white-box (partial and shadow) inference attacks including membership inference, model inversion, attribute inference, and model extraction introduced in [Liu et al. (2022)](https://www.usenix.org/conference/usenixsecurity22/presentation/liu-yugeng). For white-box membership inference, they provide 4 inputs to the attack model, which the paper says is similar to the one used by ML Privacy Meter, including the classification loss, last layer gradients, and one-hot encoding of its true label. they feed each input into a different neural network, and then concatenate the resulting embeddings as input to another neural network. Reports the F1, AUC, and accuracy as metrics. Includes differential privacy through PyTorch [Opacus](https://github.com/pytorch/opacus).
  - The tool expects a PyTorch neural network model and the data loaded using a [DataLoader](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html).
* Notes: To train the base model, shadow model, and run an attack on that dataset: `python demo.py --train_model --train_shadow`

---------------------
