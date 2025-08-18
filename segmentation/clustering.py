from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.feature_selection import VarianceThreshold
from aeon.clustering import TimeSeriesKMeans, TimeSeriesKShape, TimeSeriesKMedoids
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from data_generation.generated_data import load_new_timeseries_data, load_original_timeseries_data
from sklearn.metrics import confusion_matrix
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram



def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


def kmeans(X, y, library: str = "aeon"):
    X_rem = VarianceThreshold(threshold=(.8 * (1 - .8))).fit_transform(X)
    X_scaled = StandardScaler().fit_transform(X_rem)
    k_means = TimeSeriesKMedoids(n_clusters=5, distance="msm") #if library == "aeon" else KMeans(n_clusters=5)
    predicted_classes = k_means.fit_predict(X_scaled)

    fig, axs = plt.subplots(5, 1, figsize=(10, 8), sharex=True)

    for cluster in range(5):
        axs[cluster].set_title(f"Cluster {cluster}")
        for i, ts in enumerate(X[predicted_classes == cluster]):
            axs[cluster].plot(ts, alpha=0.5)
        axs[cluster].grid(True)

    # Confusion matrix to see what clustering does
    cm = confusion_matrix(y, predicted_classes)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted Cluster")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix: True Labels vs K-Means Clusters")
    plt.show()

    plt.tight_layout()
    plt.show()


def sklearn_hierarcichal_clustering(X, y):
    high_clustering = AgglomerativeClustering(n_clusters=5)
    predicted_classes = high_clustering.fit_predict(X)

    plt.title("Hierarchical Clustering Dendrogram")
    # plot the top three levels of the dendrogram
    plot_dendrogram(high_clustering, truncate_mode="level", p=3)
    plt.xlabel("Number of points in node (or index of point if no parenthesis).")

    # Confusion matrix to see what clustering does
    cm = confusion_matrix(y, predicted_classes)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted Cluster")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix: True Labels vs Hierarcichal Clusters")
    plt.show()

    plt.tight_layout()
    plt.show()


def svd_practice(X, y):
    svd_model = TruncatedSVD(12)
    svd = svd_model.fit_transform(X)
    print(svd_model.components_)
    print(svd_model.explained_variance_)
    print(svd_model.explained_variance_ratio_)


def main():
    og_X, og_y = load_original_timeseries_data(['mixed'], as_numpy=True)
    new_X, new_y = load_new_timeseries_data(as_numpy=True)

    X = np.concatenate((og_X, new_X))
    y = np.concatenate((og_y, new_y))

    kmeans(X, y, "aeon")
    # hierarcichal_clustering(X, y)
    svd_practice(X, y)