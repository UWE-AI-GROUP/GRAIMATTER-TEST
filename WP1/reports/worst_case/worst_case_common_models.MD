# _Worst case_ MIA properties of common models

This report documents some initial experiments done to establish possible _worst case_ behaviour of a particular target model, dataset combination.

To perform these experiments, we split a dataset into two portions -- _training_ and _testing_. The target model is trained on the _training_ set and then predictive probabilities are obtained for both the _training_ and _testing_ sets.

These predictive probabilities become the input to an _attack_ model that is trained to attempt to determine which of the two sets a data point came from based only on the predictive probabilities.

This is clearly an unrealistic attack model: if the attacker knows which data was and wasn't in the training set they don't need to attack. However, it is hoped that this might give us some idea as to the intrinsic _attackability_ of a model, dataset pair. i.e. if we cannot determine whether objects are from the _training_ or _test_ sets under this scenario, it seems incredibly likely that an attacker without this information would be able to.

For each model, dataset pair, we present the AUC and a ROC curve of the attack model, as well as a plot of the predictive probabilities of the target model (separately for each class) for the training and test data. In addition we show....

## Results with default hyper-parameters

| model | scenario | type | TPR | FPR | FAR | TNR | PPV | NPV | FNR | ACC | Advantage | PLR | NLR | OR | AUC |
|----- | -------- | ---- | --- | --- | --- | --- | --- | --- | --- | --- | --------- | --- | --- | -- | --- |
 | DecisionTree | Salem1 | shadow | 1.0 | 0.0 | 0.0 | 1.0 | 1.0 | 1.0 | 0.0 | 1.0 | 1.0 | inf | 0.0 | inf | 0.9949 |
 |  | Salem2 | shadow | 0.9009 | 0.1167 | 0.0654 | 0.8833 | 0.9346 | 0.8281 | 0.0991 | 0.8947 | 0.7842 | 7.722 | 0.1122 | 68.8312 | 0.892 |
 |  | model | WorstCase | 1.0 | 1.0 | 0.3312 | 0.0 | 0.6688 | nan | 0.0 | 0.6688 | 0.0 | 1.0 | nan | nan | 0.5 |
 |  |  | model | 1.0 | 0.0 | 0.0 | 1.0 | 1.0 | 1.0 | 0.0 | 1.0 | 1.0 | inf | 0.0 | inf | 0.9949 |
 | GaussianProcess | Salem1 | shadow | 0.0 | 0.0 | nan | 1.0 | nan | 0.3158 | 1.0 | 0.3158 | 0.0 | nan | 1.0 | nan | 0.5 |
 |  | Salem2 | shadow | 0.9459 | 0.1 | 0.0541 | 0.9 | 0.9459 | 0.9 | 0.0541 | 0.9298 | 0.8459 | 9.4595 | 0.0601 | 157.5 | 0.5713 |
 |  | model | WorstCase | 1.0 | 1.0 | 0.3312 | 0.0 | 0.6688 | nan | 0.0 | 0.6688 | 0.0 | 1.0 | nan | nan | 0.5 |
 |  |  | model | 1.0 | 0.0 | 0.0 | 1.0 | 1.0 | 1.0 | 0.0 | 1.0 | 1.0 | inf | 0.0 | inf | 0.9949 |
 | KN | Salem1 | shadow | 0.978 | 0.1667 | 0.0729 | 0.8333 | 0.9271 | 0.9459 | 0.022 | 0.9323 | 0.8114 | 5.8681 | 0.0264 | 222.5 | 0.9531 |
 |  | Salem2 | shadow | 0.964 | 0.1667 | 0.0855 | 0.8333 | 0.9145 | 0.9259 | 0.036 | 0.9181 | 0.7973 | 5.7838 | 0.0432 | 133.75 | 0.9473 |
 |  | model | WorstCase | 1.0 | 1.0 | 0.3312 | 0.0 | 0.6688 | nan | 0.0 | 0.6688 | 0.0 | 1.0 | nan | nan | 0.4872 |
 |  |  | model | 0.989 | 0.1071 | 0.0476 | 0.8929 | 0.9524 | 0.974 | 0.011 | 0.9586 | 0.8819 | 9.2308 | 0.0123 | 750.0 | 0.9903 |
 | MIA | model | Salem1 | 0.2778 | 0.5849 | 0.6739 | 0.4151 | 0.3261 | 0.3607 | 0.7222 | 0.3458 | 0.3071 | 0.4749 | 1.7399 | 0.273 | 0.3506 |
 |  |  | Salem2 | 0.675 | 0.8235 | 0.3415 | 0.1765 | 0.6585 | 0.1875 | 0.325 | 0.5263 | 0.1485 | 0.8196 | 1.8417 | 0.4451 | 0.4493 |
 | MLP | Salem1 | shadow | 0.978 | 0.0714 | 0.0326 | 0.9286 | 0.9674 | 0.9512 | 0.022 | 0.9624 | 0.9066 | 13.6923 | 0.0237 | 578.5 | 0.9895 |
 |  | Salem2 | shadow | 0.964 | 0.1167 | 0.0614 | 0.8833 | 0.9386 | 0.9298 | 0.036 | 0.9357 | 0.8473 | 8.2625 | 0.0408 | 202.5357 | 0.9807 |
 |  | model | WorstCase | 0.8692 | 1.0 | 0.363 | 0.0 | 0.637 | 0.0 | 0.1308 | 0.5812 | 0.1308 | 0.8692 | inf | 0.0 | 0.4184 |
 |  |  | model | 1.0 | 0.0 | 0.0 | 1.0 | 1.0 | 1.0 | 0.0 | 1.0 | 1.0 | inf | 0.0 | inf | 0.9949 |
 | RandomForest | Salem1 | shadow | 1.0 | 0.0 | 0.0 | 1.0 | 1.0 | 1.0 | 0.0 | 1.0 | 1.0 | inf | 0.0 | inf | 0.9949 |
 |  | Salem2 | shadow | 0.982 | 0.0333 | 0.018 | 0.9667 | 0.982 | 0.9667 | 0.018 | 0.9766 | 0.9486 | 29.4595 | 0.0186 | 1580.5 | 0.9859 |
 |  | model | WorstCase | 1.0 | 1.0 | 0.3312 | 0.0 | 0.6688 | nan | 0.0 | 0.6688 | 0.0 | 1.0 | nan | nan | 0.5 |
 |  |  | model | 1.0 | 0.0 | 0.0 | 1.0 | 1.0 | 1.0 | 0.0 | 1.0 | 1.0 | inf | 0.0 | inf | 0.9949 |
 | SVM | Salem1 | shadow | 0.978 | 0.1548 | 0.0681 | 0.8452 | 0.9319 | 0.9467 | 0.022 | 0.9361 | 0.8233 | 6.3195 | 0.026 | 243.0385 | 0.9874 |
 |  | Salem2 | shadow | 0.991 | 0.2333 | 0.1129 | 0.7667 | 0.8871 | 0.9787 | 0.009 | 0.9123 | 0.7577 | 4.2471 | 0.0118 | 361.4286 | 0.968 |
 |  | model | WorstCase | 0.6542 | 0.9623 | 0.4215 | 0.0377 | 0.5785 | 0.0513 | 0.3458 | 0.45 | 0.3081 | 0.6799 | 9.1636 | 0.0742 | 0.2337 |
 |  |  | model | 1.0 | 0.119 | 0.0521 | 0.881 | 0.9479 | 1.0 | 0.0 | 0.9624 | 0.881 | 8.4 | 0.0 | inf | 0.9949 |

## Small hyper-parameter changes can make models highly disclosive
