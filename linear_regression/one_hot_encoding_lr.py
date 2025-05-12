import pandas as pd
from sklearn import linear_model

df = pd.read_csv("linear_regression/carprices.csv")
X = df[['Car Model', 'Mileage', 'Age(yrs)']]
y = df[['Sell Price($)']]

dummies = pd.get_dummies(X[['Car Model']])
X = pd.concat([X, dummies], axis='columns')
X = X.drop(['Car Model', 'Car Model_Audi A5'], axis='columns')

reg = linear_model.LinearRegression()
reg.fit(X, y)
print(reg.predict([[45000, 4, False, True]]))
print(reg.predict([[86000, 7, True, False]]))
print(reg.score(X, y))