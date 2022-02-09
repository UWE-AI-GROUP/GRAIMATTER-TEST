# GRAIMatter Glossary

## Introduction

This document is etc etc

## TREs and actors

**TRE**: a trusted research environment....

**Researcher**: someone who has permission to use a TRE and has access to sensitive, disclosive data within that TRE. For the purposes of this work, researchers are assumed to be interested in building machine learning (ML) models that they will then wish to remove from the TRE.

**Attacker**: an individual (or organisation) who attempts to extract, from the trained model, some of the sensitive and disclosive data that was used to train it.


## Attack types etc

**Black box**: in a black box attack, the _attacker_ has query access to the model. That is, they can present input data to the model and observe the predictive outputs that the model makes. For example, in a model that detects the presence / absence of a tumour in an x-ray image, the attacker can present an image to the model and will receive the _probabilities_ that a tumour is present or not. Black box attacks do not have access to the interior of the model.

**Grey Box**: this is a variant of a black box attack where  the _attacker_  has access to the trained model in some form. For example, they have downloaded it, rather than being restricted to querying a model hosted elsewhere via a web form. Although the intention would be only to enable them to make queries by calling the model's "predict" functions (as per black box ), having access to the  model will often allow a different form of query reveal details of the models architecture or training parameters. In some cases this information can be used to inform the attack process.

**White box**:

## Disclosure Risks ##

**Membership inference** is the risk that an attacker (of whatever coloured box) can create systems that identify whether a given data point was part of the data used to train the released model.  This risk is far more likely to be disclosive of personal information in cases of medical data (_X was part of a trial for a new cancer drug_)  than  it is for other forms of data TREs might hold (_Y was part of a survey on educational outcomes_).

**Individual Disclosure** occurs when outputs from an analysis segment the participants in such a way that one sub-group has only a few members (_mean income of left-handed vegetarian professors of underwater knitting was £Y_).
 - This is especially risky if external factors might make it reasonable for one person who knew they were part of a small group to identify the others.
 - For traditional statistical analysis, where outputs might be tables designed by the researcher, a common threshold rule might be "don't release cells with less that 3 people's data in".
 - The creation of AI models automates an equivalent process to the researcher hand-designing a table,  so the same risks apply.    _It remains to be seen exactly how the _traditional_ disclosure rules relate to membership inference_.

**Group Disclosure** occurs when an analysis reveals that an assertion can be made about all members of a certain sub-group. Whether or not this is an issue may depend on the kind of data that the TRE holds.


## Metrics and technical terms ##

**Mechanism** is a term describing a procedure which takes a dataset and outputs some information about it. Usually, this is random-valued and considered for fixed data. As an example, given data $D=(X_1, X_2, X_3)$, a mechanism $M$ might return $M(D)=(X_1+X_2+X_3)/3 + \lambda$, where $\lambda$ is a random variable. 

**Differential Privacy** is a measure assigned to an output mechanism roughly stating how similar outputs can be when their training data differs by only a single sample. Limiting differential privacy ensures that a malicious attacker with access to all-but-one of the training samples would have difficulty inferring the values of the last training sample. For a full definition and treatment see [The Algorithmic Foundations of Differential Privacy](https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf), particularly chapters 2 and 3. Differential privacy can be quantified as
 - $\epsilon$-differential privacy: the probability/density of seeing some output of the model _never_ changes by more than a factor of $e^{\epsilon}$ when changing one sample 
 - ($\epsilon,\delta$)-differential privacy: the probability/density of seeing some output of the model _usually_ does not by more than a factor of $e^{\epsilon}$ when changing one sample, except for a set of values which have probability $\delta$ of being observed.
The lower $\epsilon$ and $\delta$, the more private the mechanism. Typically, lower $\epsilon$ and $\delta$ correspond to less _useful_ mechanisms, in that they contain less useful information about the true dataset.


## Specific machine learning methods ##

**Supervised learning**

**Unsupervised learning**

**Linear model**

**Neural network** including
 - recurrent
 - convolutional
 - basic MLP
 - autoencoder
 - GAN

**Decision tree**
 - random forest
 - boosting

## Trained ML behaviours ##
**Overfitting** Models that remember and model the training data too well and do not generalise well for unseen data. They can be more prone to membership attacks. 
Typically, small training datasets can lead to overfitted models, especially if the data points have many features. A bad choice of hyperparameters can also lead to overfitting (for example, excessively big NN for simple classification tasks can lead to overfitting).

Methods to reduce overfitting include increasing the training dataset size, possibly using *data augmentation techniques* and optimizing the choice of hyperparameters, possibly with cross validation. In neural networks, it is often beneficial to include *dropout* layers, which randomly deactivate neurons during training (effectively making the training procedure more noisy) and to include regularization methods (which add additional terms to the loss function that penalize overfitting). Differentially private optimizers (such as DP-SGD) add noise during the optimization steps and often lead to better generalization.

*Data augmentation techniques* generate training samples from existing samples. In the case of images, a typical technique is to resize and rotate images in the original training set to generate new samples.

# Attack schematic

SR: just putting this here for now -- it needs a lot of work, but it's to help discussions in WP1.

<img src="attacks.svg">
