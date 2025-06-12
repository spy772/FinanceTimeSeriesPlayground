Guessing model performance
==========================

Scikit Learn Models
-------------------

* Linear Regression
    - Likely to be quite innacurate with noisy data
        - Could potentially work better with PCA to reduce dimensionality
    - Only best fits with relatively straight lines, minimal distortion
    - Very quick and lightweight model
    - Hypothesis: overall 4/10 option

* Stochastic Gradient Descent
    - Same issues as Linear Regression, but might fit more closely to the data since it is a more complicated algorithm
    - Hypothesis: overall 5/10 option

* Logistic Regeression
    - Meant for binary data, it should not work well for our needs
    - Need to investigate why it sometimes functions for our datasets
    - Also lightweight and quick
    - Hypothesis: overall 2/10 option

* Descision Trees
    - Requires little data preparation (good for our needs)
    - Compute cost is proportional to how many data points there are
    - White-box model, so you can understand why it makes certain decisions unlike black-box models (like deep learning)
    - Prone to overfitting (decently likely for our dataset given we only use certain value ranges and noise levels)
    - Unstable and thus prone to massive changes in algorithm given small data changes (possibly)
    - Can create biased trees if some classes dominate the others (not an issue for our training dataset since it is mostly even)
    - Hypothesis: Overall 7/10 model

* Support Vector Machines
    - Effective in high-D spaces (ours isn't really high-D, it's 12D)
    - Effective in cases where dimension is greater than feature amount
    - Memory efficient
    - Picks out regions based on the clustering of the data samples and classifies that region as a class, seems appropriate for our needs
    - Hypothesis: Overall 8/10

* Random Forest
    - A heavier-weight but more advanced version of decision trees
        - Is an ensemble method
    - Decreases variance and mitigates overfitting that decision trees might have
    - Hypothesis: Overall 8/10

* Naive Bayes
    - Assumes each data point in an observation is independent from all the others
    - Theoretically this shouldn't be plausible because we are trying to predict an end target that is dependent on the specific sequence of datapoints
    - Since we are violating the assumption of independence among all features, this should not be effective
    - Sometimes naive bayes works even in this case, and our data isn't *necessarily* dependent, so it possibly could work with better understanding of what dependent and independent features mean, especially in time-series data
    - Hypothesis: Overall score of 4/10

* K-Nearest Neighbours
    - In cases where data is not uniformly sampled, Radius-Nearest Neighbours/Classifier might be a better choice
        - The way it works is that it has a majority vote from other datapoints in some radius $r$, instead of a majority vote from the nearest $k$ datapoints
    - It seems that KNN Classification is best for data that is discrete, while KNN Regression is better for data that is continuous
        - Thus, it could be useful for our next task in regression
    - Inefficient in higher dimensions due to the curse of dimensionality
    - Theoretically from my understanding, KNN might be a decent choice because it would highly depend on the scale of the values in order to project it onto a higher dimesion, and then it is classified by those datapoints surrounding it
    - Hypothesis: Overall score of 7.5/10

* Can PCA be used?
    - Theoretically, if we can get PCA to work on the first few (or one) component, and then on a decent few of the last ones, it could make for more models to work appropriately because in the end this is where the majority of the important data lies
    - Technically we could classify these data using solely the first, and last five data points
    - Could experiment with this one day to see if it enhances our classification


Aeon Classification Models
--------------------------

* Random Forest (aeon)
    - Interval-based approach; looks at phase-dependent intervals of the full seires
        - Calculates summary statistics (summarizing I'd assume) from the selected subseries
    - Has a few adjustments compared to the Time Series Forest (below) model
        - Features extracted from intervals generated from additional representations in periodigram (what is this?) and 1st order differences (seems like it calculates a differnece between each susbequent datapoint)
    - Seems like a more timeseries-specific Random Forest model compared to sklearn's model, so theoretically it should work better
    - Does take up more time and compute power overall I believe
    - Hypothesis: Overall score of 9/10

* Time Series Forest
    - Same interval-based details as RSTSF
    - Is an ensemble of tree classifiers built on the summary statistics of *randomly* selected intervals
        - Ironic since RSTSF also uses randomly-selected intervals
    - From each interval, mean, standard deviation, and slope are extracted from each time series and placed into a feature vector
    - These features are used to build a tree, which are added to the ensemble of models
        - Seems like classification is based off of the overall classification from all of the ensembled trees
    - Hypothesis: Overall score of 8.5/10

* K-Nearest Neighbours (aeon)
    - A Distance-based approach, where it determines the distance between one time series with another(s), in order to measure the similarity between time series
        - Something called "Dynamic Time Warping" is the best known elastic measure (whatever elastic measure is)
    - Has a feature (most likely sklearn does too) to put a weight on neighbours nearer to the observation you are trying to predict
    - Otherwise, works very similarly to the sklearn model through using `aeon's`  distance calculation
        - Thus, it should be better overall than sklearn's (theoretically)
    - Hypothesis: Overall score of 8.5/10

* Ordinal
    - A form of Dictionary-based time series classification; the premise is to turn the time series into a sequence of discrete "words" (more-so characters that have meaning to the model, not necessarily humans), and then uses the distribution of the words to classify an observation
    - Dictionary-based models form a classifier by:
        - Extracting subseries/windows from a time series
        - Transforming each window of numerical values into a discrete-valued "word"
            - Again, a word is just a *sequence* (aeon's words) of symbols (characters) over a fixed alphabet
        - Builds a sparse (aka mostly 0s) feature vector of histograms (need to understand what these are a little more) of word counts
        - Uses a classification method from their "machine learning repetoire" on the feature vectors
            - They never specify which one, but it most likely is whichever model you use
    - This specific model uses parameter selection to build ensemble members using Gaussian processes which predict MAE (mean absolute error) values for specific parameter configurations that the model has
        - A whole load of black-box math essentially
    - The best-performing members are then selected to create the final ensemble
        - Fitting this involves finding $n$ histograms, where $n$ is the time series length
        - Prediction uses 1-NN with a "histogram intersection distance function" (essentially, a distance calculation based on the feature vector mentioned earlier)
    - Honestly, it seems extremely unclear to know how effective this kind of model is without good knowledge on bag-of-words models and possibly transformer algorithms
    - Hypothesis: Overall 5/10 because of the unclarity

* Shapelet
    - Shaplets are a subseries of a time series in the training dataset
        - Specifically a subseries that is useful in discriminating classes
    - Can use `aeon.visualisation.ShapeletVisualizer` to visualize these shapelets
    - Uses Euclidean distance from the shapelets to each time series to then classify...
        - Need to research this more, the math seems rather complicated

* Feature-based (Catch22 specifically)
    - Feature-based classifiers (aeon's at least) use transformers to turn the time series into a feature vector, then placing that vector into a classifier
        - The transformers extract descriptive statistics as the features from a time series
    - This specific model uses 22 "highly-comparative time series analysis" features that are deemed to be the most discriminatory of the full set
        - This starts by getting rid of any features that are sensitive to mean and variance
        - Then it performs feature evaluation based on predictive performance, and eliminates features that don't meet a certain threshold
        - The final portion is a hierarchical clustering performed on a correlation matrix to remove redundancy of features, to finally generate 22 clusters
            - From each of these clusters, a single feature is selected by taking into account: balanced accuracy, computational efficiency, and interpretability (how it does all this it doesn't explain fully, but mentions that it takes into account "basic statistics, linear correlations, and entropy")
        - With these chosen features, a decision tree classifier is built after applying a transformation to each time series
    - In the end, it uses an `sklearn` Random Forest Classifier, unless you explicitly state you want to use a different model
    - Seems quite thorough and sounds promising, but same as Dictionary-based models it is hard to guage without thorough understanding of transformers and other advanced topics related to this field
    - Hypothesis: Overall 5/10 score because of non-understandability of how it all works (but does sound more promising overall compared to Dictionary-based classifiers to be honest...)

* Deep Learning
    - 

* HIVECOTE2 (Ensemble/Hybrid)
    - 