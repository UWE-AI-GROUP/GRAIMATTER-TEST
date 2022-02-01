# _Worst case_ MIA properties of common models

This report documents some initial experiments done to establish possible _worst case_ behaviour of a particular target model, dataset combination.

To perform these experiments, we split a dataset into two portions -- _training_ and _testing_. The target model is trained on the _training_ set and then predictive probabilities are obtained for both the _training_ and _testing_ sets.

These predictive probabilities become the input to an _attack_ model that is trained to attempt to determine which of the two sets a data point came from based only on the predictive probabilities.

This is clearly an unrealistic attack model: if the attacker knows which data was and wasn't in the training set they don't need to attack. However, it is hoped that this might give us some idea as to the intrinsic _attackability_ of a model, dataset pair. i.e. if we cannot determine whether objects are from the _training_ or _test_ sets under this scenario, it seems incredibly likely that an attacker without this information would be able to.

For each model, dataset pair, we present the AUC and a ROC curve of the attack model, as well as a plot of the predictive probabilities of the target model (separately for each class) for the training and test data. In addition we show....

## Results with default hyper-parameters

## Small hyper-parameter changes can make models highly disclosive