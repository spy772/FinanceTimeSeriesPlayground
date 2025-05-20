from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
print(df.head())

# Asked to drop sepal observations
df.drop(["sepal length (cm)", "sepal width (cm)"], axis='columns', inplace=True)

# Plot observations to see if scaling is required
plt.scatter(df["petal length (cm)"], df["petal width (cm)"])
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.show()