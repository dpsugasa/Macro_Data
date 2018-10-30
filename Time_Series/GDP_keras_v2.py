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
from keras.layers import LSTM
import credentials

#np.random.seed(123)


'''
Create Features
'''
#ISM Manufacturing PMI Composite Index
start_date = '01/01/1980'
vendor_ticker = 'ISM/MAN_PMI'

df = quandl.get(vendor_ticker, start_date = start_date)
df = df.rename(columns = {'Index':'PMI'})
df['PMI_3m'] = df['PMI'].rolling(window=3).mean()
df['PMI_6m'] = df['PMI'].rolling(window=6).mean()

#Adjusted Retail & Food Services Sales Total Seasonally Adjusted
start_date = '01/01/1980'
vendor_ticker1 = 'FRED/RSAFS'

df2 = quandl.get(vendor_ticker1, start_date = start_date)
df2 = df2.rename(columns = {'Value':'Retail_Sales'})
df2['RS_Shift'] = df2['Retail_Sales'].diff(12) #use 12m differencing to remove trend
df2['RS_Shift_3m'] = df2['RS_Shift'].rolling(window=3).mean()
df2['RS_Shift_6m'] = df2['RS_Shift'].rolling(window=6).mean()

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
df3['AS_Shift'] = df3['Auto_Sales'].diff(12) #use 12m differencing to remove trend
df3['AS_Shift_3m'] = df3['AS_Shift'].rolling(window=3).mean()
df3['AS_Shift_6m'] = df3['AS_Shift'].rolling(window=6).mean()


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
df4['IP_Shift_3m'] = df4['IP_Shift'].rolling(window=3).mean()
df4['IP_Shift_6m'] = df4['IP_Shift'].rolling(window=6).mean()

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
df5['LMCI_3m'] = df5['LMCI'].rolling(window=3).mean()
df5['LMCI_6m'] = df5['LMCI'].rolling(window=6).mean()

#US Heavy Truck Sales
mgr = dm.BbgDataManager()
# set dates, securities, and fields
IDs4 = ['USASHVTK Index']
sids4 = mgr[IDs4]
fields4 = ['LAST PRICE']

df7 = sids4.get_historical(fields4, start_date, end_date)
df7.columns = df7.columns.droplevel(-1)
df7  = df7.tshift(-1,freq='MS')
df7 = df7.rename(columns = {'USASHVTK Index':'Heavy Trucks'})
df7['HeavyT_Shift'] = df7['Heavy Trucks'].diff(12)
df7['HT_3m'] = df7['Heavy Trucks'].rolling(window=3).mean()
df7['HT_6m'] = df7['Heavy Trucks'].rolling(window=6).mean()

#US Consumer Confidence
mgr = dm.BbgDataManager()
# set dates, securities, and fields
IDs5 = ['CONCCONF Index']
sids5 = mgr[IDs5]
fields5 = ['LAST PRICE']

df8 = sids5.get_historical(fields5, start_date, end_date)
df8.columns = df8.columns.droplevel(-1)
#df8 = df8.resample('MS').mean() #not sure this is the best way to do this
df8  = df8.tshift(-1,freq='MS')
df8 = df8.rename(columns = {'CONCCONF Index':'Con_Conf'})
df8['CC_3m'] = df8['Con_Conf'].rolling(window=3).mean()
df8['CC_6m'] = df8['Con_Conf'].rolling(window=6).mean()
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
df9  = df9.tshift(-1,freq='MS')
df9 = df9.rename(columns = {'PCE CYOY Index':'Core_PCE'})
df9['CPCE_3m'] = df9['Core_PCE'].rolling(window=3).mean()
df9['CPCE_6m'] = df9['Core_PCE'].rolling(window=6).mean()
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
df10  = df10.tshift(-1,freq='MS')
df10 = df10.rename(columns = {'USDPCSAM Index':'Disp_Income'})
df10['Disp_Shift'] = df10['Disp_Income'].diff(12)
df10['DI_3m'] = df10['Disp_Shift'].rolling(window=3).mean()
df10['DI_6m'] = df10['Disp_Shift'].rolling(window=6).mean()

frames = [df, df2, df3, df4, df5, df7, df8, df9]
df_in = pd.concat(frames, join='outer', axis=1)
#df_in[['Disp_Income', 'Disp_Shift', 'DI_3m', 'DI_6m']] = \
#    df_in[['Disp_Income', 'Disp_Shift', 'DI_3m', 'DI_6m']].fillna(df10['Disp_Shift'].mean())
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
df_total = df_total.drop(['GDP','Retail_Sales', 'IP','Heavy Trucks', #'Disp_Income',
                          'Auto_Sales'], axis=1)

#scaler = pre.MinMaxScaler(feature_range = (0,1))
#data_transform = scaler.fit_transform(df_total.values)
X, y = df_total.values[:, 0:24], df_total.values[:, 24]
y = y.reshape(-1,1)
scalerx = pre.MinMaxScaler(feature_range=(0,1)).fit(X)
x_scale = scalerx.transform(X)
scalery = pre.MinMaxScaler(feature_range=(0,1)).fit(y)
y_scale = scalery.transform(y)

train_split = int(len(df_total)*0.6)
#train, test = df_total.values[0:train_split], df.total.values[train_split:]

#X,y = data_transform[:, 0:24], data_transform[:,24]
X_train, X_test = x_scale[0:train_split], x_scale[train_split:]
y_train, y_test = y_scale[0:train_split], y_scale[train_split:]
#X = np.reshape(X, (X.shape[0], 1, X.shape[1]))
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
#y_train = np.reshape(y_train, (y_train.shape[0], 1, 1))
#y_test = np.reshape(y_test, (y_test.shape[0], 1, 1))




##Create X and y; create train and test sets
#X = np.array(df_total[['PMI', 'PMI_3m', 'PMI_6m', 'RS_Shift', 'RS_Shift_3m', 'RS_Shift_6m',
#       'AS_Shift', 'AS_Shift_3m', 'AS_Shift_6m', 'IP_Shift', 'IP_Shift_3m',
#       'IP_Shift_6m', 'LMCI', 'LMCI_3m', 'LMCI_6m', 'HeavyT_Shift', 'HT_3m',
#       'HT_6m', 'Con_Conf', 'CC_3m', 'CC_6m', 'Core_PCE', 'CPCE_3m', 'CPCE_6m',
#       ]]) #Disp_Shift', 'DI_3m', 'DI_6m
#scaler = pre.MinMaxScaler(feature_range = (0,1))
#X_booty = scaler.fit_transform(X)
#y = np.array(df_total['GDP_Shift']).reshape(-1,1)
#y_booty = scaler.transform(y)
#
#train_split = int(len(X)*0.8)
#X_train, X_test = X_booty[0:train_split],X_booty[train_split:]
##X_train = X_train.astype('float32')
#y_train, y_test = y_booty[0:train_split],y_booty[train_split:]

model = Sequential()
model.add(LSTM(32, input_shape = (1,24))) #activation='relu'#model.add(Dense(8))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train, y_train, epochs=100, batch_size=2, verbose=2)

#Estimate model performance
#trainScore = model.evaluate(X_train, y_train, verbose=0)
#print ('train Score: %.2f MSE (%.2f RMSE)' % (trainScore, sqrt(trainScore)))
#testScore = model.evaluate(X_test, y_test, verbose=0)
#print ('Test Score: %.2f MSE (%.2f RMSE)' % (testScore, sqrt(testScore)))
#fullsetScore = model.evaluate(X_booty,y, verbose=0)
#print ('Full_Set_Score: %.2f MSE (%.2f RMSE)' % (fullsetScore, sqrt(fullsetScore)))

#generate predictions for training
trainPredict = scalery.inverse_transform(model.predict(X_train))
testPredict = scalery.inverse_transform(model.predict(X_test))
trainScore = model.evaluate(scalery.inverse_transform(y_train), trainPredict, verbose=0)
print ('train Score: %.2f MSE (%.2f RMSE)' % (trainScore, sqrt(trainScore)))
testScore = model.evaluate(scalery.inverse_transform(y_test), testPredict, verbose=0)
print ('Test Score: %.2f MSE (%.2f RMSE)' % (testScore, sqrt(testScore)))
##fullPredict = model.predict(X_booty)
#testPredict_plot = np.empty_like(y)
#testPredict_plot[:, :] = np.nan
#testPredict_plot[len(trainPredict):len(y),:] = testPredict
#
#plt.style.use('seaborn-bright')
#plt.figure(figsize=(10,6))
#plt.plot(y)
#plt.plot(trainPredict)
#plt.plot(testPredict_plot)
##plt.plot(fullPredict)
#plt.show()
