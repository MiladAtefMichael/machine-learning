# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 02:39:45 2020

@author: Milad
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import joblib

# Importing the dataset
dataset = pd.read_csv('C:/Users/Milad/Desktop/burn_calories_dataset.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values
advices =dataset.iloc[:, 0].values
# Training the Linear Regression model on the whole dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)



# Visualising the Linear Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()




# Predicting a new result with Polynomial Regression
predict=int(lin_reg.predict([[1000]]))
print (predict)
cal_advices = advices[predict-1]
print(cal_advices)
#save model 
filename1 = 'cal_lin_model.sav'
joblib.dump(lin_reg, filename1)


#save data
data = np.asarray(advices)
np.savetxt('cal_data.csv', data, delimiter=',',fmt='%s')