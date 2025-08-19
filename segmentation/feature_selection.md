Research on Feature Selection
=============================

Feature selection is a method to figure out which variables are most important to use in a machine learning task, and is an umbrella term for numerous feature selection methods. They fall into the usual supervised and unsupervised categories.

* A popular unsupervised method is PCA, and the way most unsupervised feature selection methods work is by selecting or combining features that have the most variance in a dataset
    - However it is not always the case that data with high variance is data that affects model performance/interpretability
    - It is also difficult to interpret the features selected, especially if they are combined together to form a new component (like in PCA), given the unsupervised nature of the methods
    - See the following Stack Overflow thread for more details https://stats.stackexchange.com/questions/514225/pca-versus-other-ways-of-feature-selections
* Supervised methods select features that are best for a *specific* task, like the retirement customer segmentation that might be done with our work
    - One important thing to remember with supervised segmentation, it is best not to perform feature selection "too early", which may incur bias in the training at test data
    - "Too early" usually means we perform feature selection on all of the input data, instead of only on the training data
    - It is best to leave the test data unchanged so that model evaluation can appropriately occur

Another key piece of research found in https://stats.stackexchange.com/questions/27300/using-principal-component-analysis-pca-for-feature-selection is that methods like PCA, SVD, or univariate screening methods (like a t-test, correlation test, etc.) do not take into account **multivariate** natures within data
    - Besides, PCA is typically only used for dimension reduction, not for feature extraction; these are two different processes
    - PCA will produce a linear combination of features that have the highest variance for the first principal component, then the second one with be the highest variance linear combination that is orthogonal to the first, and so on and so forth
    - But again, variance does not mean high correlation

## Feature Selection in Scikit-Learn

Scikit-Learn offers various feature selection algorithm types:
* Low-variance Removal **(Unsupervised)**
    - Simply removes features that have lower variance, which is not necessarily a good metric of correlation
    - Use it when a quick and basic cleanup is needed, especially to eliminate constants that aren't useful (like features that have the same value almost all the time), and you do not care much about the correlation between the features and target value (if any)
* Univariate **(Supervised)**
    - Uses statistical tests on each feature, selecting only the top `k` features
    - Statistical tests can include false-positive rates, false-discovery rates, family-wise error, chi-squared, ANOVA, F-regression, and more
    - Scikit-Learn offers a GenericUnivariateSelect class which lets the user create their own formula or feature selector
    - Useful if you want just a simple, interpretable way to rank features on their individual relevance to the target value, or even as a pre-processing step in pipelines or modelling workflows
* Recursive
    - Works by using similar scoring methods as Univariate methods, but instead it removes 1 feature every iteration and then re-scores the remaining features (using a predictive model) against each other until there are only the desired amount of features left
    - Useful when your model performance matters most, and want a high-quality feature set that is useful specifically for that model
        - As a bonus, it is able to capture feature interaction quite well
    - However, it is computational heavy, can overfit to the estimator's biases, and highly-correlated features can confuse the selector (it is best to remove redundancy or multicollinearity beforehand)
* SelectFromModel
    - L1 and Tree-based methods exist, where they fit a LinearSVC or Tree-based classifier on the data and extract the most important features after the model was trained
    - This is more akin to Permutation Feature Importance, where the most relevant features are only selected after a model is trained
    - You need to be able to set your own thresholds to do this, so it is an additional hyperparameter to tune
* Sequential
    - A greedy algorithm that either goes forwards or backwards
    - Forward starts with 0 features, and iteratively finds the best-fitting feature onto an estimator (maximizes cross-validation score), and then continues the process adding features over and over until the desired amount is reached
    - Backwards starts with all features and greedily removes features from the set, need to see how this is different from a Recursive method
    - Risk of having a local optima due to the greedy nature of the algorithm, but is typically a little more efficient than Recursive methods while still considering inter-feature interactions
    - Useful when considering a moderate amount of features and when you want to consider feature importance sequentially instead of all at once