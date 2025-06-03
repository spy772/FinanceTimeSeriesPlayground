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


# Pre-process for plotting new timeseries data in order to confirm what shape it has
columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
reg_x, reg_y = load_timeseries_data(as_numpy=False)
print(reg_x[18:24])
df = pd.DataFrame(reg_x[18:24])
df.columns = columns

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

# Plot
plot_multiple_timeseries(df)
