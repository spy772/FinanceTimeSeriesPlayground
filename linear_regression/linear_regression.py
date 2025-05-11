import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

df = pd.read_csv("linear_regression/canada_per_capita_income.csv")

plt.xlabel('year')
plt.ylabel('per capita income (US$)')
plt.scatter(df['year'], df['per capita income (US$)'], c='r', marker="+")
plt.show(block=False)
plt.pause(0.001)

reg = linear_model.LinearRegression()
reg.fit(df[['year']], df[['per capita income (US$)']])
print(reg.coef_)
print(reg.intercept_)
print(reg.predict([[2020]]))

plt.show()