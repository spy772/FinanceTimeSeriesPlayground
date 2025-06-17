from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from aeon.classification.hybrid import HIVECOTEV2
from aeon.classification.distance_based import KNeighborsTimeSeriesClassifier
from aeon.classification.interval_based import RSTSF
from aeon.classification.deep_learning import LITETimeClassifier
from aeon.classification.feature_based import Catch22Classifier
from aeon.classification.interval_based import TimeSeriesForestClassifier
from aeon.classification.shapelet_based import LearningShapeletClassifier
from aeon.classification.ordinal_classification import IndividualOrdinalTDE
from data_generation.generated_data import load_timeseries_data

# Generate monthly dates for one year
dates = pd.date_range(start='2024-01-01', periods=12, freq='MS')

# Random monthly expense values between 5 and 100

# INCREASING
expenses1 = [12.55, 15.67, 19.92, 24.31, 28.77, 33.48, 39.11, 44.6, 50.89, 56.24, 63.75, 68.93]
expenses2 = [17.28, 22.04, 25.67, 23.11, 32.8, 36.14, 41.92, 48.26, 51.3, 57.98, 64.25, 70.84]
expenses3 = [9.83, 18.27, 28.39, 36.92, 45.73, 52.1, 57.35, 60.72, 64.48, 67.12, 69.88, 72.47]
expenses4 = [14.16, 15.95, 18.42, 22.83, 28.47, 36.18, 43.92, 52.76, 64.11, 72.58, 81.37, 89.95]
expenses5 = [21.49, 26.73, 23.88, 32.65, 38.47, 43.92, 40.73, 49.27, 57.84, 64.38, 68.92, 74.16]

increasing = [expenses1, expenses2, expenses3, expenses4, expenses5]

# DECREASING
expenses6 = [92.38, 88.47, 83.92, 79.64, 76.28, 70.95, 66.12, 61.43, 58.97, 54.28, 51.36, 47.19]
expenses7 = [89.24, 85.67, 78.93, 82.46, 74.81, 70.12, 65.49, 60.87, 55.38, 50.91, 46.78, 43.25]
expenses8 = [84.56, 78.21, 80.42, 71.39, 76.33, 66.81, 69.12, 58.63, 61.18, 50.72, 48.91, 41.37]
expenses9 = [91.73, 85.49, 78.95, 73.84, 68.92, 72.65, 64.39, 60.18, 55.04, 48.37, 44.62, 38.75]
expenses10 = [87.92, 83.46, 80.17, 76.32, 71.48, 67.29, 61.74, 58.91, 54.23, 49.78, 44.32, 40.15]

decreasing = [expenses6, expenses7, expenses8, expenses9, expenses10]

# ABOUT SAME
expenses11 = [23.16, 27.42, 21.89, 25.77, 22.94, 26.18, 24.36, 29.61, 22.15, 27.39, 24.92, 23.83]
expenses12 = [34.18, 28.74, 44.93, 21.67, 39.88, 32.15, 17.94, 29.61, 36.77, 23.48, 31.59, 41.22]
expenses13 = [53.91, 59.14, 57.26, 52.48, 56.37, 54.19, 58.64, 51.73, 57.92, 53.36, 55.08, 56.89]
expenses14 = [68.44, 74.39, 70.28, 65.93, 72.16, 66.78, 69.92, 73.41, 71.23, 67.45, 70.86, 68.97]
expenses15 = [63.81, 78.49, 54.92, 69.34, 61.77, 45.26, 74.53, 58.39, 70.11, 52.73, 66.84, 59.47]

neutral = [expenses11, expenses12, expenses13, expenses14, expenses15]

mixed = [expenses1, expenses11, expenses12, expenses6, expenses2, expenses13, expenses7, expenses3, expenses4, expenses14, expenses5, expenses15, expenses8, expenses9, expenses10]
# mixed_answers = [1,3,3,2,1,3,2,1,1,3,1,3,2,2,2]
mixed_answers = [0,2,2,4,0,2,4,0,0,2,0,2,4,4,4] # For aeon training

test1 = [18.35, 21.42, 26.79, 31.67, 37.58, 42.91, 49.24, 55.87, 61.33, 68.94, 75.62, 81.37]
test2 = [23.71, 27.84, 33.62, 29.45, 38.97, 44.13, 48.36, 53.91, 60.22, 66.38, 70.15, 76.49]
test3 = [87.94, 81.45, 76.32, 71.88, 67.24, 61.93, 58.12, 52.79, 47.36, 42.85, 38.21, 33.97]
test4 = [81.33, 77.48, 70.12, 75.29, 67.93, 62.37, 66.28, 57.94, 53.12, 49.86, 44.25, 40.81]
test5 = [41.62, 38.19, 44.75, 35.84, 46.27, 39.92, 42.51, 36.78, 40.93, 45.16, 39.24, 43.67]
test6 = [42.15, 58.37, 26.94, 36.81, 45.67, 33.42, 51.93, 28.74, 39.26, 47.58, 34.91, 41.67]

test = [test1, test2, test3, test4, test5, test6]
test_answers = [1,1,2,2,3,3]

# model = LogisticRegression(max_iter=10000)
# model = tree.DecisionTreeClassifier()
combined = mixed + test
combined_answers = mixed_answers + test_answers
mixed_np = np.asarray(mixed, dtype=np.float32)
test_np = np.asarray(test, dtype=np.float32)
combined_np = np.asarray(combined, dtype=np.float32)

mixed_ans_np = np.asarray(mixed_answers, dtype=np.int32)
test_ans_np = np.asarray(test_answers, dtype=np.int32)
combined_ans_np = np.asarray(combined_answers, dtype=np.int32)

# Other things I have done
X, y = load_timeseries_data(as_numpy=True)
X = np.concatenate((X, mixed_np))
y = np.concatenate((y, mixed_ans_np))
# Without normalization, differnt scales of the data will render innacurate models
X_norm = normalize(X, norm='l2') # L2 Normalization to have scaled data be within range of 0-1

X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.3)

# Initialize various models, append them all to a list that will be used to run on the data
model_list = []
model_dict = dict()

lin_regress = LinearRegression() # Linear Regression
model_list.append(lin_regress)
model_dict["linear_regression"] = lin_regress
stochastic_gd = SGDClassifier() # Stochastic Gradient Descent
model_list.append(stochastic_gd)
model_dict["stochastic_gradient_descent"] = stochastic_gd
log_regress = LogisticRegression() # Good ol' logistic regression
model_list.append(log_regress)
model_dict["logistic_regression"] = log_regress
decision_tree = DecisionTreeClassifier() # Decision Trees
model_list.append(decision_tree)
model_dict["decision_tree"] = decision_tree
svc = SVC() # Support vector macines
model_list.append(svc)
model_dict["support_vector_machine"] = svc
random_forest = RandomForestClassifier() # Random Forest
model_list.append(random_forest)
model_dict["random_forest"] = random_forest
naive_bayes = GaussianNB() # Naive Bayes
model_list.append(naive_bayes)
model_dict["naive_bayes"] = naive_bayes
knn = KNeighborsClassifier() # KNN
model_list.append(knn)
model_dict["k_nearest_neighbours"] = knn

# Aeon models
rstsf = RSTSF() # Interval-based model
model_list.append(rstsf)
model_dict["rstsf"] = rstsf
time_series_forest = TimeSeriesForestClassifier() # Interval-based mode
model_list.append(time_series_forest)
model_dict["time_series_forest"] = time_series_forest
k_neighbours = KNeighborsTimeSeriesClassifier() # Distance-based model
model_list.append(k_neighbours)
model_dict["k_neighbours"] = k_neighbours
ordinal = IndividualOrdinalTDE() # Ordinal-based model
model_list.append(IndividualOrdinalTDE)
model_dict["ordinal"] = ordinal
shapelet = LearningShapeletClassifier() # Shapelet-based model
model_list.append(shapelet)
model_dict["shapelet"] = shapelet
catch22 = Catch22Classifier() # Feature-based model
model_list.append(catch22)
model_dict["catch22"] = catch22
# Heavier models start here
deep_learning = LITETimeClassifier() # Deep-learning-based model
model_list.append(deep_learning)
model_dict["deep_learning"] = deep_learning
hive_cote_v2 = HIVECOTEV2() # Ensemble/Hybrid model
model_list.append(hive_cote_v2)
model_dict["hive_cote_v2"] = hive_cote_v2

# Iterate through all the models
for model_name, model in model_dict.items():
    print(f"Running model: {model_name} \n")
    model.fit(X_train, y_train)
    print(f"Model score for {model_name}: {model.score(X_test, y_test)}")
    print(f"Cross Val Score for {model_name}: {model_name, cross_val_score(model, X_norm, y, cv=7)} \n\n")
    print("---------- \n\n")



# -----MANUAL TESTNG AND VISUALS -------


predict_array_increase = [[63.71, 47.84, 63.62, 69.45, 58.97, 44.13, 78.36, 53.91, 80.22, 66.38, 82.15, 91.49]]
predict_array_decrease = [[93.71, 47.84, 63.62, 89.45, 58.97, 44.13, 48.36, 33.91, 50.22, 26.38, 12.15, 21.49]]
predict_array_same = [[63.71, 47.84, 63.62, 89.45, 28.97, 44.13, 58.36, 43.91, 60.22, 56.38, 52.15, 61.49]]

predict_array = predict_array_increase
predict_np = np.asarray(predict_array, dtype=np.float32)

print('Predict: ', model.predict(predict_np))


# Create a DataFrame
# df = pd.DataFrame({'Expense': predict_array[0]}, index=dates)
columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
df = pd.DataFrame(neutral)
df.columns = columns

# Plot the line chart
# plt.figure(figsize=(10, 5))
# plt.plot(df.index, df['Expense'], marker='o', linestyle='-', color='teal')
# plt.title('Monthly Expenses for 1 Year')
# plt.xlabel('Month')
# plt.ylabel('Expense ($)')
# plt.grid(True)
# plt.xticks(df.index, df.index.strftime('%b'), rotation=45)
# plt.tight_layout()
# plt.show()

def plot_timeseries(df: pd.DataFrame | pd.Series):
    plt.figure(figsize=(10, 5))
    print(df)
    plt.plot(df.index, df, marker='o', linestyle='-', color='teal')
    plt.title('Monthly Expenses for 1 Year')
    plt.xlabel('Month')
    plt.ylabel('Expense ($)')
    plt.grid(True)
    # plt.xticks(df.index, df.index.__str__(), rotation=45)
    plt.tight_layout()
    plt.show()

def plot_multiple_timeseries(df: pd.DataFrame):
    # Create a figure and a set of subplots
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(10, 8))

    # # Plot data on the subplots
    axes[0, 0].plot(df.columns, df.iloc[0])
    axes[0, 0].set_title('List 1')

    axes[0, 1].plot(df.columns, df.iloc[1])
    axes[0, 1].set_title('List 2')

    axes[1, 0].plot(df.columns, df.iloc[2])
    axes[1, 0].set_title('List 3')

    axes[1, 1].plot(df.columns, df.iloc[3])
    axes[1, 1].set_title('List 4')

    axes[2, 0].plot(df.columns, df.iloc[4])
    axes[2, 0].set_title('List 5')

    axes[2, 1].plot(df.columns, df.iloc[5])
    axes[2, 1].set_title('List 6')

    # Remove the last subplot if needed (only want to show 5)
    # fig.delaxes(axes[2,1])

    # Adjust layout
    plt.tight_layout()

    # Show the plot
    plt.show()

# print(df.iloc[0])
# plot_timeseries(df.iloc[0])
plot_multiple_timeseries(df)
