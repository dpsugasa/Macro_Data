# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 12:41:35 2018

@author: dpsugasa
"""

#import all modules
import pandas as pd
from pandas.tseries.offsets import *
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing as pre
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
#from statsmodels.tsa.stattools import adfuller
#from statsmodels.graphics.tsaplots import plot_acf
#sfrom statsmodels.graphics.tsaplots import plot_pacf
import datetime #for dates
from datetime import datetime
import quandl #for data
from math import sqrt
import tia.bbg.datamgr as dm
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import MaxPooling2D
import credentials

np.random.seed(123)


'''
Create Features
'''
#ISM Manufacturing PMI Composite Index
start_date = '01/01/1980'
vendor_ticker = 'ISM/MAN_PMI'

df = quandl.get(vendor_ticker, start_date = start_date)
df = df.rename(columns = {'Index':'PMI'})

#Adjusted Retail & Food Services Sales Total Seasonally Adjusted
start_date = '01/01/1980'
vendor_ticker1 = 'FRED/RSAFS'

df2 = quandl.get(vendor_ticker1, start_date = start_date)
df2 = df2.rename(columns = {'Value':'Retail_Sales'})
df2['RS_Shift'] = df2['Retail_Sales'].diff(12) #use 12m differencing to remove trend

#Motor Vehicle Retail Sales: Domestic and Foreign Autos Seasonally Adjusted Annualized Rate
mgr = dm.BbgDataManager()
# set dates, securities, and fields
start_date = '01/01/1980'
end_date = "{:%m/%d/%Y}".format(datetime.now())
IDs3 = ['SAARTOTL Index']
sids3 = mgr[IDs3]
fields3 = ['LAST PRICE']

df3 = sids3.get_historical(fields3, start_date, end_date)
df3  = df3.tshift(-1,freq='MS')
df3.columns = df3.columns.droplevel(-1)
df3 = df3.rename(columns = {'SAARTOTL Index':'Auto_Sales'})
#df3['AS_Shift'] = df3['Auto_Sales'].diff(12) #use 12m differencing to remove trend

#US Industrial Production Seasonally Adjusted
mgr = dm.BbgDataManager()
# set dates, securities, and fields
start_date = '01/01/1980'
end_date = "{:%m/%d/%Y}".format(datetime.now())
IDs = ['IP Index']
sids = mgr[IDs]
fields = ['LAST PRICE']

df4 = sids.get_historical(fields, start_date, end_date)
df4  = df4.tshift(-1,freq='MS')
df4.columns = df4.columns.droplevel(-1)
df4 = df4.rename(columns = {'IP Index':'IP'})
df4['IP_Shift'] = df4['IP'].diff(12) #use 12m differencing to remove trend

#FED Average Monthly Change in Fed Labor Market Index
mgr = dm.BbgDataManager()
# set dates, securities, and fields
IDs1 = ['LMCILMCC Index']
sids1 = mgr[IDs1]
fields1 = ['LAST PRICE']

df5 = sids1.get_historical(fields1, start_date, end_date)
df5  = df5.tshift(-1,freq='MS')
df5.columns = df5.columns.droplevel(-1)
df5 = df5.rename(columns = {'LMCILMCC Index':'LMCI'})

#SPX YoY returns
mgr = dm.BbgDataManager()
# set dates, securities, and fields
IDs4 = ['SPX Index']
sids4 = mgr[IDs4]
fields4 = ['LAST PRICE']

df7 = sids4.get_historical(fields4, start_date, end_date)
df7.columns = df7.columns.droplevel(-1)
df7 = df7.resample('MS').mean() #not sure this is the best way to do this
#df7  = df7.tshift(-1,freq='MS')
df7 = df7.rename(columns = {'SPX Index':'SPX'})
df7['SPX_Shift'] = df7['SPX'].diff(12)

#US Consumer Confidence
mgr = dm.BbgDataManager()
# set dates, securities, and fields
IDs5 = ['CONCCONF Index']
sids5 = mgr[IDs5]
fields5 = ['LAST PRICE']

df8 = sids5.get_historical(fields5, start_date, end_date)
df8.columns = df8.columns.droplevel(-1)
#df8 = df8.resample('MS').mean() #not sure this is the best way to do this
#df7  = df7.tshift(-1,freq='MS')
df8 = df8.rename(columns = {'CONCCONF Index':'Con_Conf'})
#df7['SPX_Shift'] = df7['SPX'].diff(12)

#US Real PCE YoY % Change Seasonally Adjusted
mgr = dm.BbgDataManager()
# set dates, securities, and fields
IDs6 = ['PCE CYOY Index']
sids6 = mgr[IDs6]
fields6 = ['LAST PRICE']

df9 = sids6.get_historical(fields6, start_date, end_date)
df9.columns = df9.columns.droplevel(-1)
#df8 = df8.resample('MS').mean() #not sure this is the best way to do this
#df7  = df7.tshift(-1,freq='MS')
df9 = df9.rename(columns = {'PCE CYOY Index':'Core_PCE'})
#df7['SPX_Shift'] = df7['SPX'].diff(12)

#US Real Disposable Personal Income Billion Chained 2009 Dollars - SAAR Monthly
mgr = dm.BbgDataManager()
# set dates, securities, and fields
IDs7 = ['USDPCSAM Index']
sids7 = mgr[IDs7]
fields7 = ['LAST PRICE']

df10 = sids7.get_historical(fields7, start_date, end_date)
df10.columns = df10.columns.droplevel(-1)
#df8 = df8.resample('MS').mean() #not sure this is the best way to do this
#df7  = df7.tshift(-1,freq='MS')
df10 = df10.rename(columns = {'USDPCSAM Index':'Disp_Income'})
df10['Disp_Shift'] = df10['Disp_Income'].diff(12)

frames = [df, df2, df3, df4, df5, df7, df8, df9]
df_in = pd.concat(frames, join='outer', axis=1)
df_in = df_in.fillna(method = 'ffill')
df_in = df_in.dropna()

'''
Create Ouput
'''

#US GDP Nominal Dollars SAAR; quarterly data using monthly interpolations
mgr = dm.BbgDataManager()
# set dates, securities, and fields
IDs2 = ['GDP CUR$ Index']
sids2 = mgr[IDs2]
fields2 = ['LAST PRICE']

df6 = sids2.get_historical(fields2, start_date, end_date)
df6 = df6.resample('M').interpolate(method='linear')
df6  = df6.tshift(-1,freq='MS')
df6.columns = df6.columns.droplevel(-1)
df6 = df6.rename(columns = {'GDP CUR$ Index':'GDP'})
df6['GDP'] = df6['GDP'].shift(-1)
df6['GDP_Shift'] = df6['GDP'].diff(12) #use 12m differencing to remove trend
#df6['GDP_Shift_2'] = df6['GDP_Shift'].diff(12)
#df6['GDP_Shift'] = df6['GDP'].shift(-1)

#Combine 2 datasets
frames_2 = [df_in, df6]

df_total = pd.concat(frames_2, join='outer', axis=1)
df_total = df_total.dropna()
df_total = df_total.drop(['GDP','Retail_Sales', 'IP','SPX'], axis=1)

#Create X and y; create train and test sets
X = np.array(df_total[['PMI','RS_Shift','Auto_Sales','IP_Shift','LMCI','SPX_Shift', 'Con_Conf',
                       'Core_PCE']])
scalerx = pre.StandardScaler().fit(X)
X_booty = scalerx.transform(X)
y = np.array(df_total['GDP_Shift']).reshape(-1,1)
#scalery = pre.StandardScaler().fit(y)
#y_booty = scalery.transform(y)

train_split = int(len(X)*0.8)
X_train, X_test = X_booty[0:train_split],X_booty[train_split:]
#X_train = X_train.astype('float32')
y_train, y_test = y[0:train_split],y[train_split:]

model = Sequential()
model.add(Dense(32, input_shape = (8,) , activation='relu'))
#model.add(Dense(8))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train, y_train, epochs=200, batch_size=2, verbose=2)

#Estimate model performance
trainScore = model.evaluate(X_train, y_train, verbose=0)
print ('train Score: %.2f MSE (%.2f RMSE)' % (trainScore, sqrt(trainScore)))
testScore = model.evaluate(X_test, y_test, verbose=0)
print ('Test Score: %.2f MSE (%.2f RMSE)' % (testScore, sqrt(testScore)))
fullsetScore = model.evaluate(X_booty,y, verbose=0)
print ('Full_Set_Score: %.2f MSE (%.2f RMSE)' % (fullsetScore, sqrt(fullsetScore)))

#generate predictions for training
trainPredict = model.predict(X_train)
testPredict = model.predict(X_test)
fullPredict = model.predict(X_booty)
testPredict_plot = np.empty_like(y)
testPredict_plot[:, :] = np.nan
testPredict_plot[len(trainPredict):len(y),:] = testPredict

plt.style.use('seaborn-bright')
plt.figure(figsize=(10,6))
plt.plot(y)
plt.plot(trainPredict)
plt.plot(testPredict_plot)
#plt.plot(fullPredict)
plt.show()
