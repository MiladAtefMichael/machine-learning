# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 01:21:36 2020

@author: Milad
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import joblib

# Importing the dataset
dataset = pd.read_csv('C:/Users/Milad/Desktop/sleep_dataset.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values
advices =dataset.iloc[:, 0].values
# Training the Linear Regression model on the whole dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Training the Polynomial Regression model on the whole dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 3)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# Visualising the Linear Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Visualising the Polynomial Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising the Polynomial Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()



# Predicting a new result with Polynomial Regression
predict=int(lin_reg_2.predict(poly_reg.fit_transform([[23]])))
print (predict)
advice= advices[predict-1]
print(advice)
#save model 
filename1 = 'sleep_lin_model.sav'
joblib.dump(lin_reg_2, filename1)

filename2 = 'sleep_poly_model.sav'
joblib.dump(poly_reg, filename2)
#save data
data = np.asarray(advices)
np.savetxt('sleep_data.csv', data, delimiter=',',fmt='%s')