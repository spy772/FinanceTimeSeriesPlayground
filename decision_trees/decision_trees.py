import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv("decision_trees/titanic.csv")
print(df.head())

# Upon manual inspection, these columns don't seem to be useful for determining
# whether or not a passenger survived, or there are too many NaN values to be able
# to properly assign a proper value to the column (or the correlation doesn't seem)
# obvious
df.drop(['PassengerId','Name','SibSp','Parch','Ticket','Cabin','Embarked'],axis='columns',inplace=True)
print(df.head())

X = df.drop('Survived', axis='columns')
y = df.Survived # didn't drop in-place, so it still exists in df

# Decide which version is better later
# inputs.Sex = inputs.Sex.map({'male': 1, 'female': 2})
# pd.concat([X, pd.get_dummies(X.Sex)])
# X.drop(['Sex', 'male'], inplace=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

print(model.predict(X_test[0:5]))
print(model.score(X_test, y_test))
