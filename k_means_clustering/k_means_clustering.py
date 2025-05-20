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
# plt.show()

# Cluster data
km_model = KMeans(n_clusters=3)
y_pred = km_model.fit_predict(df)

# Place predicted values in df
df['cluster'] = y_pred # Creates new column cluster with values of y_pred
print(df.head())

# df.cluster.unique() will return an array of unique values. We know there are
# 3 clusters anyways so we will just display these 3 clusters
df0 = df[df.cluster == 0]
df1 = df[df.cluster == 1]
df2 = df[df.cluster == 2]

plt.scatter(df0['petal length (cm)'],df0['petal width (cm)'],color='blue')
plt.scatter(df1['petal length (cm)'],df1['petal width (cm)'],color='green')
plt.scatter(df2['petal length (cm)'],df2['petal width (cm)'],color='yellow')
# plt.show()

### Elbow plot ###
sum_squared_errors = []
k_range = range(1, 10)
for k in k_range:
    km_model = KMeans(n_clusters=k)
    km_model.fit(df)
    sum_squared_errors.append(km_model.inertia_) # Inertia is sse of clustering

# Plot elbow plot
plt.xlabel('K Value')
plt.ylabel('Sum of Square Error (SSE)')
plt.plot(k_range, sum_squared_errors)
# plt.show()