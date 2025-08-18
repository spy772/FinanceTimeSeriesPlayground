Customer Segmentation Technical Research
========================================

## Common Methods for this task

* RFM Analysis
    - Stands for Recency, Frequency, Monetary
    - Segment customers based on their financial behaviour
    - Useful when you have transactional history, need quick and interpreteable insights, and are doing simple customer lifestyle analysis
        - Mostly for business-related problems, not very technical
* K-Means Clustering **(unsupervised)**
    - Create clusters based on a mathematical algorithm that finds `k` centroids and then clusters the nearest observations (taking into account all features in the vector space) around those centroids
    - Is simple to understand and is decently quick, but you need to know how to pick the right `k` value (of which there isn't a known way to just pick it correctly; you must be satified with the clusters visually/through observation, or by using the elbow method) and is sensitive to scaling
        - There is a general guideline in picking an optimal `k` value
        - The **elbow method** is a way to pick this value, but you can only use this if you are able to iteratively increasing `k` and running K-Means Clustering over and over
        - Afterwards, you create a plot of the errors of each `k` value and then pick the value just after the sharp decrease ends near the start of the plot
        - There is definitely a better way to use this method too, look over CISC 271 videos on K-Means Clustering and/or PCA to remember how to use the elbow method without having to run K-Means so often
    - Useful when we have structured numeric data (like demographics and behaviour), and we want automated group discovery
        - Ideally, there is a nearly even amount of observations in each cluster once clustered
* Hierarchical Clustering **(unsupervised)**
    - Builds a dendrogram/tree of clusters on its own, and does not need a specific `k` value
    - It finds the amount of clusters based on other factors (need to research), but especially similarity (like all clustering algorithms)
    - This method is extremely computationally heavy for large datasets
    - Useful when we want interpretable cluster hierarachy, and we have a smaller/medium-sized dataset
* DBSCAN (Density-Based Clustering) **(unsupervised)**
    - Groups observations based on their density in the vector space
    - Quite useful for trying to find outliers and irregularily shaped segments in the overarching data
    - This method needs carefuly tuning of hyperparameters
    - Useful when we have irregular cluster shapes or noise, want to exclude outliers, and want to be flexible with the amount of clusters generated
* Gaussian Mixture Models **(unsupervised)**
    - Purely probabilistic clustering, where each cluster is modelled as a normal distribution (bell curve)
    - However, this method is very theoretical and scientific, and it assumes the data is already normally distributed; overall a complex method
    - Useful when we want "soft clustering" (not sure what this is)
* Dimensionality Reduction into Clustering Methods **(unsupervised)**
    - Usually only used for high-dimensional data
    - Tradeoff is higher computational speed for slightly lower accuracy
        - But this isn't necessarily the case all the time, sometimes you may even be more accurate by removing some features if they aren't relevant
        - Usually the tradeoff is quite worth it
    - PCA (Principal Components Analysis) is a very popular and well-known method
        - Revise mathematics in CISC 271 to understand PCA using the SVD (singular vector decomposition), an incredibly powerful technique that shows high understanding of PCA and vector spaces
    - t-SNE and UMAP are techniques used for visualization primarily, but PCA can help with it too
* Supervised Segmentation **(supervised)**
    - If we already have labelled segments (e.g. low income, high income, etc.) then we can use classification techniques to cluster the data
        - Logistic regression for binary, decision trees/random forest for multiclass, gradient boosting for high dimension data, etc.
    - This is no longer a clustering problem, since it is now supervised and inherintely classification
* Behavioural + Psychographic Segmentation
    - Includes a much more manual approach to clustering the data, since it is more qualitative-based
    - Examples include website clickstream data, surveys, NLP on reviews and social media
        - More human behavioural-based than anything, and is not fully ML-integrated


## Good practices to remember when doing segmentation tasks

* Scale the features before clustering, so that they are either normalized or standardized
* Use domain-specific knowledge to engineer specific features in the data (especially useful for churn risk types of problems, like in the Data Science for Business book)
* Use the elbow method (learned in CISC 271) to choose the best number of clusters
    - Or, use the silhouette method (need to learn)
    - Silhouette is used when it is ambiguous to decide on a proper value when using the elbow method (according to a Medium article, not sure how legitimate this is)
* Ensure the clusters are interpretable after they are settled