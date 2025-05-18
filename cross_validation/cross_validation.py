from sklearn.datasets import load_iris
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np

iris = load_iris()

lr = cross_val_score(LogisticRegression(), iris.data, iris.target, cv=15)
svm = cross_val_score(SVC(), iris.data, iris.target, cv=15)
rfc = cross_val_score(RandomForestClassifier(n_estimators=40), iris.data, iris.target, cv=15)
dtc = cross_val_score(DecisionTreeClassifier(), iris.data, iris.target, cv=15)

print(np.average(lr))
print(np.average(svm))
print(np.average(rfc))
print(np.average(dtc))