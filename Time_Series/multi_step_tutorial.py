# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 23:20:07 2018

@author: dpsugasa
"""

import numpy as np
import pandas as pd
import pandas_datareader.data as web
from datetime import datetime

from sklearn import preprocessing
 
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.recurrent import LSTM
from keras.layers import  Dropout
 
import matplotlib.pyplot as plt
 
start = datetime(2013, 2, 9)
end = datetime(2018, 5, 24)
data_csv = web.DataReader('AAPL', 'iex', start, end)
##fname = "C:\\Users\\dpsugasa\\WorkFiles\\Macro_Data\\Book1.csv"
#fname="http://intelligentonlinetools.com/blog/stock-data-file/"
#data_csv = pd.read_html(fname)
# 
#index_col = 'Date'
#how many data we will use 
# (should not be more than dataset length )
data_to_use= 100
 
# number of training data
# should be less than data_to_use
train_end =70
 
 
total_data=len(data_csv)
 
#most recent data is in the end 
#so need offset
start=total_data - data_to_use
 
 
#currently doing prediction only for 3 steps ahead
steps_to_predict = 3
train_mse=[]
test_mse=[]
forecast=[]
for i in range(steps_to_predict):
    train_mse.append(0)
    test_mse.append(0)
    forecast.append(0)
 
 
yt = data_csv.iloc [start:total_data ,3]    #Close price
yt1 = data_csv.iloc [start:total_data ,0]   #Open
yt2 = data_csv.iloc [start:total_data ,1]   #High
yt3 = data_csv.iloc [start:total_data ,2]   #Low
vt = data_csv.iloc [start:total_data ,4]    # volume
 
 
     
for i in range(steps_to_predict):
     
    if i==0:
        units=20
        batch_size=1
    if i==1:
        units=15
        batch_size=1
    if i==2:
         units=80
         batch_size=1
     
         
 
    yt_ = yt.shift (-i - 1  )   
 
    data = pd.concat ([yt, yt_, vt, yt1, yt2, yt3], axis =1)
    data. columns = ['yt', 'yt_', 'vt', 'yt1', 'yt2', 'yt3']
     
    data = data.dropna()
     
    
     
# target variable - closed price
    y = data ['yt_']
 
        
#       closed,  volume,   open,  high,   low    
    cols =['yt',    'vt',  'yt1', 'yt2', 'yt3']
    x = data [cols]
 
   
    
    scaler_x = preprocessing.MinMaxScaler ( feature_range =( -1, 1))
    x = np. array (x).reshape ((len( x) ,len(cols)))
    x = scaler_x.fit_transform (x)
 
    
    scaler_y = preprocessing. MinMaxScaler ( feature_range =( -1, 1))
    y = np.array (y).reshape ((len( y), 1))
    y = scaler_y.fit_transform (y)
 
 
 
     
    x_train = x [0: train_end,]
 
 
    x_test = x[ train_end +1:len(x),]    
    y_train = y [0: train_end] 
 
 
 
    y_test = y[ train_end +1:len(y)]  
                
    if (i == 0) :     
        prediction_data=[]
        for j in range (len(y_test) - 0 ) :
               prediction_data.append (0)       
 
 
 
    x_train = x_train.reshape (x_train. shape + (1,)) 
    x_test = x_test.reshape (x_test. shape + (1,))
 
     
     
 
     
 
 
    seed =2018
    np.random.seed (seed)
     
##############
##  i=0
##############
    if i == 0 :
          fit0 = Sequential ()
          fit0.add (LSTM (  units , activation = 'tanh', inner_activation = 'hard_sigmoid' , input_shape =(len(cols), 1) ))
          fit0.add(Dropout(0.2))
          fit0.add (Dense (output_dim =1, activation = 'linear'))
          fit0.compile (loss ="mean_squared_error" , optimizer = "adam")  
    
          fit0.fit (x_train, y_train, batch_size =batch_size, nb_epoch =25, shuffle = False)
          train_mse[i] = fit0.evaluate (x_train, y_train, batch_size =batch_size)
          test_mse[i] = fit0.evaluate (x_test, y_test, batch_size =batch_size)
          pred = fit0.predict (x_test) 
          pred = scaler_y.inverse_transform (np. array (pred). reshape ((len( pred), 1)))
             # below is just fo i == 0
          for j in range (len(pred) - 0 ) :
                   prediction_data[j] = pred[j] 
                   
              
                
          forecast[i]=pred[-1]
          
                   
#############
##  i=1
#############
    if i == 1 :    
          fit1 = Sequential ()
          fit1.add (LSTM (  units , activation = 'tanh', inner_activation = 'hard_sigmoid' , input_shape =(len(cols), 1) ))
          fit1.add(Dropout(0.2))
          fit1.add (Dense (output_dim =1, activation = 'linear'))
          fit1.compile (loss ="mean_squared_error" , optimizer = "adam")  
          fit1.fit (x_train, y_train, batch_size =batch_size, nb_epoch =25, shuffle = False)
          train_mse[i] = fit1.evaluate (x_train, y_train, batch_size =batch_size)
          test_mse[i] = fit1.evaluate (x_test, y_test, batch_size =batch_size)
          pred = fit1.predict (x_test) 
          pred = scaler_y.inverse_transform (np. array (pred). reshape ((len( pred), 1)))
          forecast[i]=pred[-1]
         
           
        
#############
##  i=2
#############
    if i==2 :
          fit2 = Sequential ()
          fit2.add (LSTM (  units , activation = 'tanh', inner_activation = 'hard_sigmoid' , input_shape =(len(cols), 1) ))
          fit2.add(Dropout(0.2))
          fit2.add (Dense (output_dim =1, activation = 'linear'))
          fit2.compile (loss ="mean_squared_error" , optimizer = "adam")  
          fit2.fit (x_train, y_train, batch_size =batch_size, nb_epoch =25, shuffle = False)
          train_mse[i] = fit2.evaluate (x_train, y_train, batch_size =batch_size)
          test_mse[i] = fit2.evaluate (x_test, y_test, batch_size =batch_size)
          pred = fit2.predict (x_test) 
          pred = scaler_y.inverse_transform (np. array (pred). reshape ((len( pred), 1)))
               
          forecast[i]=pred[-1]
  
 
    x_test = scaler_x.inverse_transform (np. array (x_test). reshape ((len( x_test), len(cols))))
    
prediction_data = np.asarray(prediction_data)
prediction_data = prediction_data.ravel()
 
 
 
for j in range (len(prediction_data) - 1 ):
    prediction_data[len(prediction_data) - j - 1  ] =  prediction_data[len(prediction_data) - 1 - j - 1]
 
 
prediction_data = np.append(prediction_data, forecast)
 
x_test_all = yt[len(yt)-len(prediction_data)-1:len(yt)-1]
x_test_all = x_test_all.ravel()                
 
plt.plot(prediction_data, label="predictions")
plt.plot(  x_test_all, label="actual")
 
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=2)
 
import matplotlib.ticker as mtick
fmt = '$%.0f'
tick = mtick.FormatStrFormatter(fmt)
ax = plt.axes()
ax.yaxis.set_major_formatter(tick)
 
 
plt.show()
 
 
print ("prediction data")
print ((prediction_data))
 
print ("x_test_all")
print ((x_test_all))
 
print ("train_mse")
print (train_mse)
 
print ("test_mse")
print (test_mse)