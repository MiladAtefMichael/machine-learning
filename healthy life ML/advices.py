# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 23:14:47 2020

@author: Milad
"""

#import librares
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
#load data for heart rate
lin_model = joblib.load('finalized_lin_model.sav')
poly_model = joblib.load('finalized_poly_model.sav')
data=pd.read_csv('data.csv')
#load data for sleep
sleep_lin_model = joblib.load('sleep_lin_model.sav')
sleep_poly_model = joblib.load('sleep_poly_model.sav')
sleep_data=pd.read_csv('sleep_data.csv')
#load data for calories 
cal_lin_model = joblib.load('cal_lin_model.sav')
cal_data=pd.read_csv('cal_data.csv')
#load data for blood pressure 
pressure_lin_model = joblib.load('pressure_lin_model.sav')
pressure_data=pd.read_csv('pressure_data.csv')


#heart rate advice
heart_advices=data.iloc[:,0]
predict=int(lin_model.predict(poly_model.fit_transform([[40]])))
print (predict)
advice= heart_advices[predict-1]
print(advice)
#sleep advice
sleep_advices=sleep_data.iloc[:,0]
sleep_predict=int(sleep_lin_model.predict(sleep_poly_model.fit_transform([[21]])))
print (sleep_predict)
sleep_advice= sleep_advices[sleep_predict-1]
print(sleep_advice)
#calories advices 
cal_advices=cal_data.iloc[:,0]
cal_predict=int(cal_lin_model.predict([[2000]]))
print (cal_predict)
cal_advice= cal_advices[cal_predict-1]
print(cal_advice)
#blood pressure advices 
pressure_advices=pressure_data.iloc[:,0]
pressure_predict=int(pressure_lin_model.predict([[120,70]]))
print (pressure_predict)
pressure_advice= pressure_advices[cal_predict-1]
print(pressure_advice)

