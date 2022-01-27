# GRAIMatter Glossary

## Introduction

This document is etc etc

## TREs and actors

**TRE**: a trusted research environment....

**Researcher**: someone who has permission to use a TRE and has access to sensitive, disclosive data within that TRE. For the purposes of this work, researchers are assumed to be interested in building machine learning (ML) models that they will then wish to remove from the TRE.

**Attacker**: an individual (or organisation) who attempts to extract, from the trained model, some of the sensitive and disclosive data that was used to train it.


## Attack types etc

**Black box**: in a black box attack, the _attacker_ has query access to the model. That is, they can present input data to the moodel and observe the predictive outputs that the model makes. For example, in a model that detects the presence / absence of a tumour in an x-ray image, the attacker can present an image to the model and will recieve the _probabilities_ that a tumour is present or not. Black box attacks do not have access to the interior of the model.