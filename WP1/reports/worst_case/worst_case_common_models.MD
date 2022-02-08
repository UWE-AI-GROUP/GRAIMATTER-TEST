# _Worst case_ MIA properties of common models

This report documents some initial experiments done to establish possible _worst case_ behaviour of a particular target model, dataset combination.

To perform these experiments, we split a dataset into two portions -- _training_ and _testing_. The target model is trained on the _training_ set and then predictive probabilities are obtained for both the _training_ and _testing_ sets.

These predictive probabilities become the input to an _attack_ model that is trained to attempt to determine which of the two sets a data point came from based only on the predictive probabilities.

This is clearly an unrealistic attack model: if the attacker knows which data was and wasn't in the training set they don't need to attack. However, it is hoped that this might give us some idea as to the intrinsic _attackability_ of a model, dataset pair. i.e. if we cannot determine whether objects are from the _training_ or _test_ sets under this scenario, it seems incredibly likely that an attacker without this information would be able to.

For each model, dataset pair, we present the AUC and a ROC curve of the attack model, as well as a plot of the predictive probabilities of the target model (separately for each class) for the training and test data. In addition we show....

## Results with default hyper-parameters

The initial models have been trained with a subset of MIMIC-II data (https://physionet.org/content/mimic2-iaccd/1.0/).

The data is split into 3: target_train (0.5), shadow_train (0.25) and test (0.25). The target_train is used to train the target model, the shadow_train data is for training Salem 1 shadow model and the test data is used to test in all cases. For Salem 2 the breast cancer dataset available from sklearn is used for the subshadow models (0.7 for training and 0.3 for testing).

All membership inference attacks use RandomForest classifier.

- Decision Tree MIA
	- Worst Case
<img src="worst_case/DecisionTree - MIA RandomForest.pdf"><img src="worst_case/DecisionTree - MIA RandomForest_class_0.pdf"><img src="worst_case/DecisionTree - MIA RandomForest_class_1.pdf">
	- Salem1
<img src="Salem1/ATTACK DecisionTree - MIA RandomForest.pdf"><img src="Salem1/ATTACK DecisionTree - MIA RandomForest_class_0.pdf"><img src="Salem1/ATTACK DecisionTree - MIA RandomForest_class_1.pdf">
	- Salem2
<img src="Salem2/Salem-adv2  DecisionTree - MIA RandomForest.pdf"><img src="Salem2/Salem-adv2  DecisionTree - MIA RandomForest_class_0.pdf"><img src="Salem2/Salem-adv2 DecisionTree - MIA RandomForest_class_1.pdf">

- GaussianProcess MIA
	- Worst Case
<img src="worst_case/GaussianProcess - MIA RandomForest.pdf"><img src="worst_case/GaussianProcess - MIA RandomForest_class_0.pdf"><img src="worst_case/DecisionTree - MIA RandomForest_class_1.pdf">
	- Salem1
<img src="Salem1/ATTACK GaussianProcess - MIA RandomForest.pdf"><img src="Salem1/ATTACK GaussianProcess - MIA RandomForest_class_0.pdf"><img src="Salem1/ATTACK DecisionTree - MIA RandomForest_class_1.pdf">
	- Salem2
<img src="Salem2/Salem-adv2 GaussianProcess - MIA RandomForest.pdf"><img src="Salem2/Salem-adv2 GaussianProcess - MIA RandomForest_class_0.pdf"><img src="Salem2/Salem-adv2 GaussianProcess - MIA RandomForest_class_1.pdf">

Table summarising different metrics. It is sorted by classifiers and then the different scenarios.

| model | scenario | type | TPR | FPR | FAR | TNR | PPV | NPV | FNR | ACC | Advantage | PLR | NLR | OR | AUC |
|----- | -------- | ---- | --- | --- | --- | --- | --- | --- | --- | --- | --------- | --- | --- | -- | --- |
 | DecisionTree | Salem1 | MIA | 1.0 | 1.0 | 0.4953 | 0.0 | 0.5047 | nan | 0.0 | 0.5047 | 0.0 | 1.0 | nan | nan | 0.5 |
 | DecisionTree | Salem1 | shadow | 1.0 | 0.0 | 0.0 | 1.0 | 1.0 | 1.0 | 0.0 | 1.0 | 1.0 | inf | 0.0 | inf | 0.9949 |
 | DecisionTree | Salem2 | MIA | 1.0 | 1.0 | 0.2982 | 0.0 | 0.7018 | nan | 0.0 | 0.7018 | 0.0 | 1.0 | nan | nan | 0.5 |
 | DecisionTree | Salem2 | shadow | 0.9009 | 0.1333 | 0.0741 | 0.8667 | 0.9259 | 0.8254 | 0.0991 | 0.8889 | 0.7676 | 6.7568 | 0.1143 | 59.0909 | 0.8837 |
 | DecisionTree | model | WorstCase | 1.0 | 1.0 | 0.3312 | 0.0 | 0.6688 | nan | 0.0 | 0.6688 | 0.0 | 1.0 | nan | nan | 0.5 |
 | DecisionTree | model | model | 1.0 | 0.0 | 0.0 | 1.0 | 1.0 | 1.0 | 0.0 | 1.0 | 1.0 | inf | 0.0 | inf | 0.9949 |
 | GaussianProcess | Salem1 | MIA | 1.0 | 0.0 | 0.0 | 1.0 | 1.0 | 1.0 | 0.0 | 1.0 | 1.0 | inf | 0.0 | inf | 0.9949 |
 | GaussianProcess | Salem1 | shadow | 0.0 | 0.0 | nan | 1.0 | nan | 0.3158 | 1.0 | 0.3158 | 0.0 | nan | 1.0 | nan | 0.5 |
 | GaussianProcess | Salem2 | MIA | 1.0 | 0.0 | 0.0 | 1.0 | 1.0 | 1.0 | 0.0 | 1.0 | 1.0 | inf | 0.0 | inf | 0.9949 |
 | GaussianProcess | Salem2 | shadow | 0.9459 | 0.1 | 0.0541 | 0.9 | 0.9459 | 0.9 | 0.0541 | 0.9298 | 0.8459 | 9.4595 | 0.0601 | 157.5 | 0.5713 |
 | GaussianProcess | model | WorstCase | 1.0 | 1.0 | 0.3312 | 0.0 | 0.6688 | nan | 0.0 | 0.6688 | 0.0 | 1.0 | nan | nan | 0.5 |
 | GaussianProcess | model | model | 1.0 | 0.0 | 0.0 | 1.0 | 1.0 | 1.0 | 0.0 | 1.0 | 1.0 | inf | 0.0 | inf | 0.9949 |
 | KN | Salem1 | MIA | 0.9434 | 0.9444 | 0.505 | 0.0556 | 0.495 | 0.5 | 0.0566 | 0.4953 | 0.001 | 0.9989 | 1.0189 | 0.9804 | 0.4736 |
 | KN | Salem1 | shadow | 0.978 | 0.1667 | 0.0729 | 0.8333 | 0.9271 | 0.9459 | 0.022 | 0.9323 | 0.8114 | 5.8681 | 0.0264 | 222.5 | 0.9531 |
 | KN | Salem2 | MIA | 1.0 | 1.0 | 0.2982 | 0.0 | 0.7018 | nan | 0.0 | 0.7018 | 0.0 | 1.0 | nan | nan | 0.5165 |
 | KN | Salem2 | shadow | 0.964 | 0.1667 | 0.0855 | 0.8333 | 0.9145 | 0.9259 | 0.036 | 0.9181 | 0.7973 | 5.7838 | 0.0432 | 133.75 | 0.9473 |
 | KN | model | WorstCase | 1.0 | 1.0 | 0.3312 | 0.0 | 0.6688 | nan | 0.0 | 0.6688 | 0.0 | 1.0 | nan | nan | 0.5137 |
 | KN | model | model | 0.989 | 0.1071 | 0.0476 | 0.8929 | 0.9524 | 0.974 | 0.011 | 0.9586 | 0.8819 | 9.2308 | 0.0123 | 750.0 | 0.9903 |
 | MLP | Salem1 | MIA | 0.9074 | 0.9057 | 0.4948 | 0.0943 | 0.5052 | 0.5 | 0.0926 | 0.5047 | 0.0017 | 1.0019 | 0.9815 | 1.0208 | 0.5178 |
 | MLP | Salem1 | shadow | 0.978 | 0.0595 | 0.0273 | 0.9405 | 0.9727 | 0.9518 | 0.022 | 0.9662 | 0.9185 | 16.4308 | 0.0234 | 703.1 | 0.991 |
 | MLP | Salem2 | MIA | 0.7875 | 0.7353 | 0.2841 | 0.2647 | 0.7159 | 0.3462 | 0.2125 | 0.6316 | 0.0522 | 1.071 | 0.8028 | 1.3341 | 0.5192 |
 | MLP | Salem2 | shadow | 0.9459 | 0.1167 | 0.0625 | 0.8833 | 0.9375 | 0.8983 | 0.0541 | 0.924 | 0.8293 | 8.1081 | 0.0612 | 132.5 | 0.9823 |
 | MLP | model | WorstCase | 0.8692 | 1.0 | 0.363 | 0.0 | 0.637 | 0.0 | 0.1308 | 0.5812 | 0.1308 | 0.8692 | inf | 0.0 | 0.4076 |
 | MLP | model | model | 1.0 | 0.0 | 0.0 | 1.0 | 1.0 | 1.0 | 0.0 | 1.0 | 1.0 | inf | 0.0 | inf | 0.9949 |
 | RandomForest | Salem1 | MIA | 1.0 | 0.037 | 0.0364 | 0.963 | 0.9636 | 1.0 | 0.0 | 0.9813 | 0.963 | 27.0 | 0.0 | inf | 0.9812 |
 | RandomForest | Salem1 | shadow | 1.0 | 0.0 | 0.0 | 1.0 | 1.0 | 1.0 | 0.0 | 1.0 | 1.0 | inf | 0.0 | inf | 0.9949 |
 | RandomForest | Salem2 | MIA | 1.0 | 0.4412 | 0.1579 | 0.5588 | 0.8421 | 1.0 | 0.0 | 0.8684 | 0.5588 | 2.2667 | 0.0 | inf | 0.7794 |
 | RandomForest | Salem2 | shadow | 0.982 | 0.05 | 0.0268 | 0.95 | 0.9732 | 0.9661 | 0.018 | 0.9708 | 0.932 | 19.6396 | 0.019 | 1035.5 | 0.9853 |
 | RandomForest | model | WorstCase | 1.0 | 1.0 | 0.3312 | 0.0 | 0.6688 | nan | 0.0 | 0.6688 | 0.0 | 1.0 | nan | nan | 0.5 |
 | RandomForest | model | model | 1.0 | 0.0 | 0.0 | 1.0 | 1.0 | 1.0 | 0.0 | 1.0 | 1.0 | inf | 0.0 | inf | 0.9949 |
 | SVM | Salem1 | MIA | 0.4259 | 0.434 | 0.5 | 0.566 | 0.5 | 0.4918 | 0.5741 | 0.4953 | 0.008 | 0.9815 | 1.0142 | 0.9677 | 0.559 |
 | SVM | Salem1 | shadow | 0.978 | 0.1548 | 0.0681 | 0.8452 | 0.9319 | 0.9467 | 0.022 | 0.9361 | 0.8233 | 6.3195 | 0.026 | 243.0385 | 0.9874 |
 | SVM | Salem2 | MIA | 0.675 | 0.6471 | 0.2895 | 0.3529 | 0.7105 | 0.3158 | 0.325 | 0.5789 | 0.0279 | 1.0432 | 0.9208 | 1.1329 | 0.4826 |
 | SVM | Salem2 | shadow | 0.991 | 0.2333 | 0.1129 | 0.7667 | 0.8871 | 0.9787 | 0.009 | 0.9123 | 0.7577 | 4.2471 | 0.0118 | 361.4286 | 0.968 |
 | SVM | model | WorstCase | 0.6916 | 0.9811 | 0.4127 | 0.0189 | 0.5873 | 0.0294 | 0.3084 | 0.4688 | 0.2895 | 0.7049 | 16.3458 | 0.0431 | 0.2474 |
 | SVM | model | model | 1.0 | 0.119 | 0.0521 | 0.881 | 0.9479 | 1.0 | 0.0 | 0.9624 | 0.881 | 8.4 | 0.0 | inf | 0.9949 |

## Small hyper-parameter changes can make models highly disclosive
