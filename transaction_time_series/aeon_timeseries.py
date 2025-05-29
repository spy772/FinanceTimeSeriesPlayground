from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from aeon.classification.hybrid import HIVECOTEV2
from aeon.classification.distance_based import KNeighborsTimeSeriesClassifier

from generate_data import load_timeseries_data

X_np, y_np = load_timeseries_data(as_numpy=True)
X_train, X_test, y_train, y_test = train_test_split(X_np, y_np, test_size=0.3)

model = KNeighborsTimeSeriesClassifier()
model.fit(X_train, y_train)

print(model.score(X_test, y_test))