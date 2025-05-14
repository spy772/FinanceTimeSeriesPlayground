from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

iris = load_iris()
print(iris.data[0])
print(iris.target[0])

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2)
model = LogisticRegression()
model.fit(X_train, y_train)

print(model.score(X_test, y_test))
print(model.predict(X_test[0:5]))

# Confusion Matrix work
y_predicted = model.predict(X_test)

cm = confusion_matrix(y_test, y_predicted)
plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')