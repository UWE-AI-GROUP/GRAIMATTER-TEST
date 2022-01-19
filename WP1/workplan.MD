# WP1 Workplan

Add tasks here, with links to github issues where appropriate

## Create a model privacy testing scaffold

Create the code necessary to be able to assess models w.r.t membership inference / model inversion. Should make it easy to switch between models and datasets.

1. Assess the suitability of privacy meter for this -- will it work for all models, or just tf NNs [looks like it only works with tensorflow, pytorch, openvino]
1. Create a 'homemade' version as per the notebook in `WP1/notebooks/` that will generate privacy-meter like outputs for sklearn models [#21](https://github.com/jim-smith/GRAIMatter/issues/21)

## Identify representative datasets

1. Identify and document a set of representative open datasets [#5](https://github.com/jim-smith/GRAIMatter/issues/5)

## Identify ML models we want to evaluate

1. Create list of ML models to evaluate [#11](https://github.com/jim-smith/GRAIMatter/issues/11)

## Test default sklearn models

Test a range of sklearn models, with default parameters to assess the privacy risk.

1. Run analysis scaffold over a series of sklearn models, assessing privacy risk with _default_ hyper-params across representative datasets.
1. Create test scaffold to explore range of hyper-params for these models.

## FAIR


	- Investigate open source implementation of FAIR for different tasks (ex: 
theonaunheim/pyfair: Factor Analysis of Information Risk (FAIR) model written in Python. (github.com))

	- Define loss, risks, factors.
	- Investigate the feasibility to include hyperparameters  as a factor
  - From experiments  results on the different chosen models investigate how to derive the loss frequency and loss magnitude


## Synthetic data

TODO: James
