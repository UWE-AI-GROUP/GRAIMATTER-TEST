# GRAIMatter Glossary

## Introduction

This document is etc etc

## TREs and actors

**TRE**: a trusted research environment....

**Researcher**: someone who has permission to use a TRE and has access to sensitive, disclosive data within that TRE. For the purposes of this work, researchers are assumed to be interested in building machine learning (ML) models that they will then wish to remove from the TRE.

**Attacker**: an individual (or organisation) who attempts to extract, from the trained model, some of the sensitive and disclosive data that was used to train it.


## Attack types etc

**Black box**: in a black box attack, the _attacker_ has query access to the model. That is, they can present input data to the moodel and observe the predictive outputs that the model makes. For example, in a model that detects the presence / absence of a tumour in an x-ray image, the attacker can present an image to the model and will recieve the _probabilities_ that a tumour is present or not. Black box attacks do not have access to the interior of the model.
 
**Grey Box**: this is a variant of a black box attack where  the _attacker_  has access to the trained model in some form. For example, they have downloaded it, rather than being restricted to querying a model hosted elsewhere via a web form. Although the intention would be only to enable them to make queries by calling the model's "predict" functions (as per black box ), having access to the  model will often allow a different form of query reveal details of the models architecture or training parameters. In some cases this information can be used to inform the attack process. 

**White box**:

## Disclosure Risks ##

**Membership inference** is the risk that an attacker (of whatever coloured box) can create systems that identify whether a given data point was part of the data used to train the released model.  This risk is far more likely to be disclosive of personal information in cases of medical data (_X was part of a trial for a new cancer drug_)  than  it is for other forms of data TREs might hold (_Y was part of a survey on educational outcomes_).

**Individual Disclosure** occurs when outputs from an analysis segment the participants in such a way that one sub-group has only a few members (_mean income of left-handed vegetarian professors of underwater knitting was £Y_). 
 - This is expecially risky if external factors might make it reasonable for one person who knew they were part of a small group to identify the others.
 - For traditional statistical analysis, where outputs might be tables designed by the reseacher, a common threshold rule might be "don't release cells with less that 3 people's data in".
 - The creation of AI models automates an equivalent process to the researcher hand-designing a table,  so the same risks apply.    _It remains to be seen exactly how the _traditional_ disclosure rules relate to membership inference_.
 
**Group Disclosure** occurs when an analysis reveals that an assertion can be made about all members of a certain sub-group. Whether or not this is an issue may depend on the kind of data that the TRE holds.


