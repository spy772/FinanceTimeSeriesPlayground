import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

df = pd.read_csv("logistic_regression/employee_retention.csv")
print(df.head())

left = df[df.left == 1]
retained = df[df.left == 0]
print(left.shape)
print(retained.shape)

print(df.groupby('left').mean(numeric_only=True))

# Shows that employees with higher salaries tend to stay 
pd.crosstab(df.salary, df.left).plot(kind='bar')

# Doesn't show much correlation to retention rate, ignore Department for analysis
pd.crosstab(df.Department,df.left).plot(kind='bar')

# Use salaries and remaining data (not Department) as data to classify with
sub_df = df[['satisfaction_level','average_montly_hours','promotion_last_5years','salary']]

# Need to one-hot encode it, prefix appends "salary_" in front of "high, mid, low"
salary_dummies = pd.get_dummies(sub_df.salary, prefix="salary")
df_with_dummies = pd.concat([sub_df, salary_dummies], axis='columns')
df_with_dummies.drop('salary', axis='columns', inplace=True)

# Data is now pre-processed
X = df_with_dummies
y = df.left

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.3)

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

print(model.predict(X_test))
print(model.score(X_test, y_test))