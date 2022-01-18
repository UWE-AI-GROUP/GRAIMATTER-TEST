#!/usr/bin/python3

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier, plot_tree

iris = datasets.load_iris()
X = iris.data
y = iris.target

# print the max and min values in each feature to help hand-craft the disclopsive point
for feature in range(4):
    print(f"feature {feature} min {np.min(X[:,feature])}, min {np.max(X[:,feature])}")


# example code with no safety
raw_dt = DecisionTreeClassifier(min_samples_leaf=1, criterion="gini", random_state=1)
raw_dt.fit(X, y)

#Jim adding code to test obfuscation
numsamples = 5
import getpass

if getpass.getuser()=="j4-smith":
    numsamples=1
raw_dt2 = DecisionTreeClassifier(min_samples_leaf=numsamples, criterion="gini", random_state=1)   
print(f'through obfuscation made a tree with {raw_dt.min_samples_leaf} samples per leaf')

print(f"Training set accuracy in this naive case is {raw_dt.score(X,y)}")

fig, ax = plt.subplots(1, 1, figsize=(15, 10))
output = plot_tree(raw_dt, filled=True, ax=ax, fontsize=11)

# Diligent user realises problem, and changes their code to enforce at least n samples per leaf
# We'll use n=5

manual_dt = DecisionTreeClassifier(min_samples_leaf=5, criterion="gini", random_state=1)
manual_dt.fit(X, y)

print(f"Training set accuracy in this naive case is {manual_dt.score(X,y)}")

fig2, ax2 = plt.subplots(1, 1, figsize=(15, 10))
output = plot_tree(manual_dt, filled=True, ax=ax2, fontsize=11)


# Assign(targets=[Name(id='rawDT', ctx=Store())],
# value=Call(func=Name(id='DecisionTreeClassifier', ctx=Load()), args=[],
# keywords=[keyword(arg='min_samples_leaf', value=Constant(value=1,
# kind=None)), keyword(arg='criterion', value=Constant(value='gini',
# kind=None)), keyword(arg='random_state', value=Constant(value=1,
# kind=None))]), type_comment=None),

# Expr(value=Call(func=Attribute(value=Name(id='rawDT', ctx=Load()),
# attr='fit', ctx=Load()), args=[Name(id='X', ctx=Load()), Name(id='y',
# ctx=Load())], keywords=[])),


# Call(func=Attribute(value=Name(id='datasets', ctx=Load()), attr='load_iris',
# ctx=Load()), args=[], keywords=[])
