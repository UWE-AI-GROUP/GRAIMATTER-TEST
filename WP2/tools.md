# Notes on open source privacy assessment tools

Are fairness/ethical "bias" checking tools relevant to this project? For example, IBM tool: https://github.com/Trusted-AI/AIF360

---------------------

"[Libraries.io](https://libraries.io) indexes data from 5,086,238 packages from 32 package managers. We monitor package releases, analyse each project's code, community, distribution and documentation, and we map the relationships between packages when they're declared as a dependency."

It provides a searchable index of packages via keyword - perhaps we can specify some keywords and then systematically go through the packages to check for relevant ones - the keywords used will help other researchers know what we searched for? List so far ["trusted-ai", "trustworthy-ai", "poisoning", "privacy-protection", "privacy-enhancing-technologies"]. Same approach applied to github search?

It generates a "source rank" for many projects, which is based on a set of criteria that include the project documentation, licenses, recency and number of versions available, whether [semantic version numbering](https://semver.org) is used, popularity measures such as github stars, the number of contributors, and how many packages or repositories are dependent on the project.

Following this general criteria, below are notes on some tools:

---------------------

## [IBM AI Privacy Toolkit](https://github.com/IBM/ai-privacy-toolkit)

MIT License;
Source code available on github; distribution available in PyPi;
Popularity: 14 stars; 3 contributors;
No tags have been generated to assign version numbers (and there is no version tagged as a release);
[Libraries.io](https://libraries.io/pypi/ai-privacy-toolkit) SourceRank: 7

Should we further inspect this?

---------------------

## [Fawkes](https://github.com/Shawn-Shan/fawkes)

BSD-3 License;
Source code available on github; distribution available in PyPi;
popularity: 4.4k stars; 3 contributors;
Single release (version 0.3 on 30 Jul, 2020);
[Libraries.io](https://libraries.io/pypi/fawkes) SourceRank: 13

Should we further inspect this?

"we propose Fawkes, a system that helps individuals to inoculate their images against unauthorized facial recognition models at any time without significantly distorting their own photos, or wearing conspicuous patches. Fawkes achieves this by helping users adding imperceptible pixel-level changes (“cloaks”) to their own photos. For example, a user who wants to share content (e.g. photos) on social media or the public web can add small, imperceptible alterations to their photos before uploading them. If collected by a third-party “tracker” and used to train a facial recognition model to recognize the user, these “cloaked” images would produce functional models that consistently misidentify them."

---------------------

## [ML-PePR](https://github.com/hallojs/ml-pepr)

GPLv3 License;
Source code available on github; distribution available in PyPi;
popularity: 0 stars; 3 contributors;
Uses semantic version numbering with 6 releases (0.1b6 latest version on 17 Jul, 2021)
[Libraries.io](https://libraries.io/pypi/mlpepr) SourceRank: 6

Should we further inspect this?

---------------------

## [ML Privacy Meter](https://github.com/privacytrustlab/ml_privacy_meter)

MIT License;
Source code available on github; no distributions available;
popularity: 241 stars; 6 contributors;
No tags have been generated to assign version numbers (and there is no version tagged as a release);

The tool expects a Keras/TensorFlow model; the dataset must be in a specific format; while the documentation states that "the API is built on top of TensorFlow 2.1 with Python 3.6", most of the tutorials and utility scripts are in antiquated Python 2 (e.g., see [here](https://github.com/privacytrustlab/ml_privacy_meter/tree/master/datasets) "so you need to have Python 2 and numpy with Python 2 support"); the tool requires the setting of a number of model/dataset specific parameters including an optimiser, a list of layers to exploit, the learning rate, number of training epochs, which means that it will likely require both (a) human involvement to define the set of parameters to explore and (b) multiple runs of the tool for tuning;

---------------------

## [Adversarial Robustness Toolbox](https://github.com/Trusted-AI/adversarial-robustness-toolbox)

MIT License;
Source code available on github; distribution available in PyPi;
popularity: 2.7k stars; 75 contributors;
27 releases using semantic version numbers (first release Apr 25, 2018; current version 1.9.1 released Jan 7, 2022);
[Libraries.io](https://libraries.io/pypi/adversarial-robustness-toolbox) SourceRank: 14
Uses continuous integration tools; unit tests; and static analysis;

6 github projects depend on the package, including [IBM AI Fairness 360](https://github.com/Trusted-AI/AIF360) and [ML-PePR](https://github.com/hallojs/ml-pepr) and [TrojAI](https://pypi.org/project/troj/1.0.0/)

"ART supports all popular machine learning frameworks (TensorFlow, Keras, PyTorch, MXNet, scikit-learn, XGBoost, LightGBM, CatBoost, GPy, etc.), all data types (images, tables, audio, video, etc.) and machine learning tasks (classification, object detection, speech recognition, generation, certification, etc.)."

---------------------

## [ML-Doctor](https://github.com/liuyugeng/ML-Doctor)


Apache-2.0 License;
Source code available on github; no distributions available;
popularity: 10 stars; 1 contributor;
No tags have been generated to assign version numbers (and there is no version tagged as a release);

---------------------

## [Diffprivlib](https://github.com/IBM/differential-privacy-library)

MIT License;
Source code available on github; distribution available in PyPi;
popularity: 555 stars; 10 contributors;
6 releases using semantic version numbers (first release Jun 19, 2019; current version 0.5.0 released Oct 1, 2021);
[Libraries.io](https://libraries.io/pypi/diffprivlib) SourceRank: 12

3 projects depend on the package: [IBM Watson](http://ibm-wml-api-pyclient.mybluemix.net/) and [synthetic-data-generation](https://github.com/daanknoors/synthetic_data_generation) and [spark-privacy-preserver](https://github.com/ThaminduR/spark-privacy-preserver)

---------------------
