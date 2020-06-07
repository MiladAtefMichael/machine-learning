# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 03:05:32 2020

@author: Milad
"""


import pandas as pd
import joblib
#load data
classifyModel=joblib.load('classify_user_model.sav')
#prepare data
burnet_cal=3000
input_cal=0
if(burnet_cal<=1000):
    input_cal=0
elif (burnet_cal>1000 and burnet_cal<2000):
    input_cal=1
else :
    input_cal=2
print(input_cal)  
up=200
down=150
blood_pressure_avr=int((up+down)/2 )
#prediction
prediction=int(classifyModel.predict([[input_cal,45,blood_pressure_avr]]) )
print(prediction)
    
        
    