# _Worst case_ MIA properties of common models

This report documents some initial experiments done to establish possible _worst case_ behaviour of a particular target model, dataset combination.

To perform these experiments, we split a dataset into two portions -- _training_ and _testing_. The target model is trained on the _training_ set and then predictive probabilities are obtained for both the _training_ and _testing_ sets.

These predictive probabilities become the input to an _attack_ model that is trained to attempt to determine which of the two sets a data point came from based only on the predictive probabilities.

This is clearly an unrealistic attack model: if the attacker knows which data was and wasn't in the training set they don't need to attack. However, it is hoped that this might give us some idea as to the intrinsic _attackability_ of a model, dataset pair. i.e. if we cannot determine whether objects are from the _training_ or _test_ sets under this scenario, it seems incredibly likely that an attacker without this information would be able to.

For each model, dataset pair, we present the AUC and a ROC curve of the attack model, as well as a plot of the predictive probabilities of the target model (separately for each class) for the training and test data. In addition we show....

## Results with default hyper-parameters

Table summarising different metrics. It is sorted by classifiers and then the different scenarios.

| model | scenario | type | TPR | FPR | FAR | TNR | PPV | NPV | FNR | ACC | Advantage | PLR | NLR | OR | AUC |
|----- | -------- | ---- | --- | --- | --- | --- | --- | --- | --- | --- | --------- | --- | --- | -- | --- |
 | DecisionTree | Salem1 | MIA | 1.0 | 1.0 | 0.5047 | 0.0 | 0.4953 | nan | 0.0 | 0.4953 | 0.0 | 1.0 | nan | nan | 0.5 |
 | DecisionTree | Salem1 | shadow | 1.0 | 0.0 | 0.0 | 1.0 | 1.0 | 1.0 | 0.0 | 1.0 | 1.0 | inf | 0.0 | inf | 0.9949 |
 | DecisionTree | Salem2 | MIA | 1.0 | 1.0 | 0.2982 | 0.0 | 0.7018 | nan | 0.0 | 0.7018 | 0.0 | 1.0 | nan | nan | 0.5 |
 | DecisionTree | Salem2 | shadow | 0.9099 | 0.1167 | 0.0648 | 0.8833 | 0.9352 | 0.8413 | 0.0901 | 0.9006 | 0.7932 | 7.7992 | 0.102 | 76.4714 | 0.8965 |
 | DecisionTree | model | WorstCase | 1.0 | 1.0 | 0.3312 | 0.0 | 0.6688 | nan | 0.0 | 0.6688 | 0.0 | 1.0 | nan | nan | 0.5 |
 | DecisionTree | model | model | 1.0 | 0.0 | 0.0 | 1.0 | 1.0 | 1.0 | 0.0 | 1.0 | 1.0 | inf | 0.0 | inf | 0.9949 |
 | GaussianProcess | Salem1 | MIA | 1.0 | 0.0 | 0.0 | 1.0 | 1.0 | 1.0 | 0.0 | 1.0 | 1.0 | inf | 0.0 | inf | 0.9949 |
 | GaussianProcess | Salem1 | shadow | 0.0 | 0.0 | nan | 1.0 | nan | 0.3158 | 1.0 | 0.3158 | 0.0 | nan | 1.0 | nan | 0.5 |
 | GaussianProcess | Salem2 | MIA | 1.0 | 0.0 | 0.0 | 1.0 | 1.0 | 1.0 | 0.0 | 1.0 | 1.0 | inf | 0.0 | inf | 0.9949 |
 | GaussianProcess | Salem2 | shadow | 0.9459 | 0.1 | 0.0541 | 0.9 | 0.9459 | 0.9 | 0.0541 | 0.9298 | 0.8459 | 9.4595 | 0.0601 | 157.5 | 0.5713 |
 | GaussianProcess | model | WorstCase | 1.0 | 1.0 | 0.3312 | 0.0 | 0.6688 | nan | 0.0 | 0.6688 | 0.0 | 1.0 | nan | nan | 0.5 |
 | GaussianProcess | model | model | 1.0 | 0.0 | 0.0 | 1.0 | 1.0 | 1.0 | 0.0 | 1.0 | 1.0 | inf | 0.0 | inf | 0.9949 |
 | KN | Salem1 | MIA | 0.1132 | 0.2407 | 0.6842 | 0.7593 | 0.3158 | 0.4659 | 0.8868 | 0.4393 | 0.1275 | 0.4702 | 1.168 | 0.4026 | 0.4299 |
 | KN | Salem1 | shadow | 0.978 | 0.1667 | 0.0729 | 0.8333 | 0.9271 | 0.9459 | 0.022 | 0.9323 | 0.8114 | 5.8681 | 0.0264 | 222.5 | 0.9531 |
 | KN | Salem2 | MIA | 1.0 | 1.0 | 0.2982 | 0.0 | 0.7018 | nan | 0.0 | 0.7018 | 0.0 | 1.0 | nan | nan | 0.516 |
 | KN | Salem2 | shadow | 0.964 | 0.1667 | 0.0855 | 0.8333 | 0.9145 | 0.9259 | 0.036 | 0.9181 | 0.7973 | 5.7838 | 0.0432 | 133.75 | 0.9473 |
 | KN | model | WorstCase | 1.0 | 1.0 | 0.3312 | 0.0 | 0.6688 | nan | 0.0 | 0.6688 | 0.0 | 1.0 | nan | nan | 0.5004 |
 | KN | model | model | 0.989 | 0.1071 | 0.0476 | 0.8929 | 0.9524 | 0.974 | 0.011 | 0.9586 | 0.8819 | 9.2308 | 0.0123 | 750.0 | 0.9903 |
 | MLP | Salem1 | MIA | 0.0926 | 0.1698 | 0.6429 | 0.8302 | 0.3571 | 0.4731 | 0.9074 | 0.4579 | 0.0772 | 0.5453 | 1.093 | 0.4989 | 0.4498 |
 | MLP | Salem1 | shadow | 0.978 | 0.0595 | 0.0273 | 0.9405 | 0.9727 | 0.9518 | 0.022 | 0.9662 | 0.9185 | 16.4308 | 0.0234 | 703.1 | 0.9892 |
 | MLP | Salem2 | MIA | 0.775 | 0.7059 | 0.2791 | 0.2941 | 0.7209 | 0.3571 | 0.225 | 0.6316 | 0.0691 | 1.0979 | 0.765 | 1.4352 | 0.4639 |
 | MLP | Salem2 | shadow | 0.9459 | 0.1 | 0.0541 | 0.9 | 0.9459 | 0.9 | 0.0541 | 0.9298 | 0.8459 | 9.4595 | 0.0601 | 157.5 | 0.9822 |
 | MLP | model | WorstCase | 0.9252 | 1.0 | 0.3487 | 0.0 | 0.6513 | 0.0 | 0.0748 | 0.6188 | 0.0748 | 0.9252 | inf | 0.0 | 0.4034 |
 | MLP | model | model | 1.0 | 0.0 | 0.0 | 1.0 | 1.0 | 1.0 | 0.0 | 1.0 | 1.0 | inf | 0.0 | inf | 0.9949 |
 | RandomForest | Salem1 | MIA | 1.0 | 0.0741 | 0.0702 | 0.9259 | 0.9298 | 1.0 | 0.0 | 0.9626 | 0.9259 | 13.5 | 0.0 | inf | 0.9628 |
 | RandomForest | Salem1 | shadow | 1.0 | 0.0 | 0.0 | 1.0 | 1.0 | 1.0 | 0.0 | 1.0 | 1.0 | inf | 0.0 | inf | 0.9949 |
 | RandomForest | Salem2 | MIA | 1.0 | 0.5294 | 0.1837 | 0.4706 | 0.8163 | 1.0 | 0.0 | 0.8421 | 0.4706 | 1.8889 | 0.0 | inf | 0.7353 |
 | RandomForest | Salem2 | shadow | 0.982 | 0.05 | 0.0268 | 0.95 | 0.9732 | 0.9661 | 0.018 | 0.9708 | 0.932 | 19.6396 | 0.019 | 1035.5 | 0.9846 |
 | RandomForest | model | WorstCase | 1.0 | 1.0 | 0.3312 | 0.0 | 0.6688 | nan | 0.0 | 0.6688 | 0.0 | 1.0 | nan | nan | 0.5 |
 | RandomForest | model | model | 1.0 | 0.0 | 0.0 | 1.0 | 1.0 | 1.0 | 0.0 | 1.0 | 1.0 | inf | 0.0 | inf | 0.9949 |
 | SVM | Salem1 | MIA | 0.5185 | 0.5849 | 0.5254 | 0.4151 | 0.4746 | 0.4583 | 0.4815 | 0.4673 | 0.0664 | 0.8865 | 1.1599 | 0.7643 | 0.5537 |
 | SVM | Salem1 | shadow | 0.978 | 0.1548 | 0.0681 | 0.8452 | 0.9319 | 0.9467 | 0.022 | 0.9361 | 0.8233 | 6.3195 | 0.026 | 243.0385 | 0.9874 |
 | SVM | Salem2 | MIA | 0.675 | 0.8235 | 0.3415 | 0.1765 | 0.6585 | 0.1875 | 0.325 | 0.5263 | 0.1485 | 0.8196 | 1.8417 | 0.4451 | 0.4427 |
 | SVM | Salem2 | shadow | 0.991 | 0.2333 | 0.1129 | 0.7667 | 0.8871 | 0.9787 | 0.009 | 0.9123 | 0.7577 | 4.2471 | 0.0118 | 361.4286 | 0.968 |
 | SVM | model | WorstCase | 0.5794 | 0.9434 | 0.4464 | 0.0566 | 0.5536 | 0.0625 | 0.4206 | 0.4062 | 0.364 | 0.6142 | 7.4299 | 0.0827 | 0.2051 |
 | SVM | model | model | 1.0 | 0.119 | 0.0521 | 0.881 | 0.9479 | 1.0 | 0.0 | 0.9624 | 0.881 | 8.4 | 0.0 | inf | 0.9949 |



## Small hyper-parameter changes can make models highly disclosive
