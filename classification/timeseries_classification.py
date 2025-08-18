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
from data_generation.generated_data import load_new_timeseries_data, load_original_timeseries_data
import datetime
import colorama
from colorama import Fore, Style

# Generate monthly dates for one year
dates = pd.date_range(start='2024-01-01', periods=12, freq='MS')

# Other things I have done
og_X, og_y = load_original_timeseries_data(['mixed'])
new_X, new_y = load_new_timeseries_data(as_numpy=True)
X = np.concatenate((og_X, new_X))
y = np.concatenate((og_y, new_y))
# Without normalization, differnt scales of the data will render innacurate models
X_norm = normalize(X, norm='l2') # L2 Normalization to have scaled data be within range of 0-1

X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.3)

# Initialize various models, append them all to a list that will be used to run on the data
model_list = []
model_dict = dict()

lin_regress = LinearRegression() # Linear Regression
model_list.append(lin_regress)
model_dict["linear_regression (scikit)"] = lin_regress
stochastic_gd = SGDClassifier() # Stochastic Gradient Descent
model_list.append(stochastic_gd)
model_dict["stochastic_gradient_descent (scikit)"] = stochastic_gd
log_regress = LogisticRegression() # Good ol' logistic regression
model_list.append(log_regress)
model_dict["logistic_regression (scikit)"] = log_regress
decision_tree = DecisionTreeClassifier() # Decision Trees
model_list.append(decision_tree)
model_dict["decision_tree (scikit)"] = decision_tree
svc = SVC() # Support vector macines
model_list.append(svc)
model_dict["support_vector_machine (scikit)"] = svc
random_forest = RandomForestClassifier() # Random Forest
model_list.append(random_forest)
model_dict["random_forest (scikit)"] = random_forest
naive_bayes = GaussianNB() # Naive Bayes
model_list.append(naive_bayes)
model_dict["naive_bayes (scikit)"] = naive_bayes
knn = KNeighborsClassifier() # KNN
model_list.append(knn)
model_dict["k_nearest_neighbours (scikit)"] = knn

# Aeon models
rstsf = RSTSF() # Interval-based model
model_list.append(rstsf)
model_dict["rstsf (aeon)"] = rstsf
time_series_forest = TimeSeriesForestClassifier() # Interval-based mode
model_list.append(time_series_forest)
model_dict["time_series_forest (aeon)"] = time_series_forest
k_neighbours = KNeighborsTimeSeriesClassifier() # Distance-based model
model_list.append(k_neighbours)
model_dict["k_neighbours (aeon)"] = k_neighbours
ordinal = IndividualOrdinalTDE() # Ordinal-based model
model_list.append(IndividualOrdinalTDE)
model_dict["ordinal (aeon)"] = ordinal

# [REMOVED] shapelet is extremely slow/resource intense and scoring results are not great average cross_val_score was 0.5070028028571428 - see test1.run for details. Timing wasn't logged, but it was about 20-30 minutes to run on M4 processor
# shapelet = LearningShapeletClassifier() # Shapelet-based model
# model_list.append(shapelet)
# model_dict["shapelet (aeon)"] = shapelet

catch22 = Catch22Classifier() # Feature-based model
model_list.append(catch22)
model_dict["catch22 (aeon)"] = catch22

# Heavier models start here

# [REMOVED] deep_learning has same issue as shapelet - average cross val score was 0.7492997171428571 and it took A LOT OF TIME to run, like 30 or 40 minutes on M4 processor. See test1.run for details
# deep_learning = LITETimeClassifier() # Deep-learning-based model
# model_list.append(deep_learning)
# model_dict["deep_learning (aeon)"] = deep_learning

hive_cote_v2 = HIVECOTEV2() # Ensemble/Hybrid model
model_list.append(hive_cote_v2)
model_dict["hive_cote_v2 (aeon)"] = hive_cote_v2
 
# Iterate through all the models
for model_name, model in model_dict.items():
    print(" \n\n----------\n\n")
    print(Fore.RED + f"Running model: {model_name} \n" + Fore.BLACK)
    totalStartTime = datetime.datetime.now()

    trainingStartTime = datetime.datetime.now()
    print(f"Single Run - Training Start time: {trainingStartTime}")
    model.fit(X_train, y_train)
    trainingEndTime = datetime.datetime.now()
    print(f"Single Run - Training End time: {trainingEndTime}")
    print(f"Single Run - Total Training time (in seconds): {(trainingEndTime - trainingStartTime).total_seconds()} \n")

    scoreStartTime = datetime.datetime.now()
    print(f"Single Run - Score Start time: {scoreStartTime}")
    print(Fore.BLUE + f"Single Run - Model score for {model_name}: {model.score(X_test, y_test)}" + Fore.BLACK)
    scoreEndTime = datetime.datetime.now()
    print(f"Single Run - Score End time: {scoreEndTime}")
    print(f"Single Run - Total Score time (in seconds): {(scoreEndTime - scoreStartTime).total_seconds()} \n")


    crossValStartTime = datetime.datetime.now()
    print(f"Cross Val - Start time: {crossValStartTime}")
    crossValResult = cross_val_score(model, X_norm, y, cv=7)
    print(Fore.YELLOW + f"Cross Val - Score: {crossValResult}" + Fore.BLACK)
    print(Fore.GREEN + f"Cross Val - Average: {sum(crossValResult) / len(crossValResult)}" + Fore.BLACK)
    crossValEndTime = datetime.datetime.now()
    print(f"Crolss Val - End time: {crossValEndTime}")
    print(f"Cross Val - Total time (in seconds): {(crossValEndTime - crossValStartTime).total_seconds()} \n")

    totalEndTime = datetime.datetime.now()
    print(f"Total time to run fit, score and cross val (in seconds): {(totalEndTime - totalStartTime).total_seconds()} \n")



# -----MANUAL TESTNG AND VISUALS -------


predict_array_increase = [[63.71, 47.84, 63.62, 69.45, 58.97, 44.13, 78.36, 53.91, 80.22, 66.38, 82.15, 91.49]]
predict_array_decrease = [[93.71, 47.84, 63.62, 89.45, 58.97, 44.13, 48.36, 33.91, 50.22, 26.38, 12.15, 21.49]]
predict_array_same = [[63.71, 47.84, 63.62, 89.45, 28.97, 44.13, 58.36, 43.91, 60.22, 56.38, 52.15, 61.49]]

predict_array = predict_array_increase
predict_np = np.asarray(predict_array, dtype=np.float32)

# print('Predict: ', model.predict(predict_np))


# Create a DataFrame
# df = pd.DataFrame({'Expense': predict_array[0]}, index=dates)
columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
df = pd.DataFrame(X)
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
# plot_multiple_timeseries(df)
