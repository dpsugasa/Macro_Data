# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 10:42:02 2018

Building a new GDP model. 
'Now Casting with Keras':  trying to predict YoY change in GDP
No effort to make the input time series stationary


@author: dpsugasa
"""

#import all modules
import pandas as pd
from pandas.tseries.offsets import *
import numpy as np
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
from keras.layers import MaxPooling2D, LSTM
import plotly
import plotly.plotly as py #for plotting
import plotly.offline as offline
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.figure_factory as ff
import credentials


booty = {'INDPRO':   'IP',               #Industrial Production
        'NETEXP':   'Exports',          #Net Exports of Goods and Services
        'EXPGSC1':  'Real_Exports',     #Real Exports of Goods and Services
        'DGORDER':  'NewOrders',        #Manufacturers' New Orders: Durable Goods
        'NEWORDER': 'NewOrders_NoDef',  #Manufacturers' New Orders: Nondefense Capital Goods Excluding Aircraft
        'PCECC96':  'Real_PCE',         #Real Personal Consumption Expenditures
        'IMPGSC1':  'Real_Imports',     #Real imports of goods and services
        'IMPGS':    'Imports',          #Imports of Goods and Services
        'AWHMAN':   'Wk_hrs_manu',      #Average Weekly Hours of Production and Nonsupervisory Employees: Manufacturing
        'AWHNONAG': 'Wk_hrs_priv',      #Average Weekly Hours of Production and Nonsupervisory Employees: Total private
        'USSLIND':  'Leading_Idx',      #Leading Index for the United States
        'TOTALSA':  'Tot_veh_sales',    #Total Vehicle Sales
        'ALTSALES': 'Lt_wght_veh',      #Light Weight Vehicle Sales: Autos and Light Trucks
        'HTRUCKSSAAR': 'Heavy_trucks',  #Motor Vehicle Retail Sales: Heavy Weight Trucks
        'FRBKCLMCIM':   'LMCI',         #KC Fed Labor Market Conditions Index, Momentum Indicator
        'UMCSENT':      'UMich',        #University of Michigan: Consumer Sentiment
        }

d = {} #dict of data

for code, name in booty.items():
    d[name] = fred.get_series_latest_release(code)
    d[name] = d[name].resample('M').last()
    d[name] = d[name].interpolate(method = 'linear')
    
frames = [d[i] for i in booty.values()]
columns = [i for i in booty.values()]

baf = pd.concat(frames, keys = columns, join = 'outer', axis = 1)
baf = baf.fillna(method = 'ffill')
baf = baf.dropna()

#ISM Manufacturing PMI Composite Index
start_date = '01/01/1980'
vendor_ticker = 'ISM/MAN_PMI'

df = quandl.get(vendor_ticker, start_date = start_date)
df = df.rename(columns = {'Index':'PMI'})
df = df.resample('M').last()

#US Consumer Confidence
mgr = dm.BbgDataManager()
# set dates, securities, and fields
start_date = '01/01/1980'
end_date = "{:%m/%d/%Y}".format(datetime.now())
IDs5 = ['CONCCONF Index']
sids5 = mgr[IDs5]
fields5 = ['LAST PRICE']

df2 = sids5.get_historical(fields5, start_date, end_date)
df2.columns = df2.columns.droplevel(-1)
#df8 = df8.resample('MS').mean() #not sure this is the best way to do this
#d27  = df7.tshift(-1,freq='MS')
df2 = df2.rename(columns = {'CONCCONF Index':'Con_Conf'})

frames2 = [baf, df, df2]
baf2 = pd.concat(frames2, join = 'outer', axis = 1)
baf2 = baf.fillna(method = 'ffill')
baf2 = baf.dropna()

'''
Create Ouput

'''
#GDP YOY
mgr = dm.BbgDataManager()
# set dates, securities, and fields
start_date = '01/01/1980'
end_date = "{:%m/%d/%Y}".format(datetime.now())
IDs6 = ['GDP CYOY Index']
sids6 = mgr[IDs6]
fields5 = ['LAST PRICE']

output = sids6.get_historical(fields5, start_date, end_date)
output.columns = output.columns.droplevel(-1)
output = output.resample('M').last() #not sure this is the best way to do this
output = output.interpolate(method = 'linear')
gdp_1 = output.shift()
gdp_2 = output.shift(2)
gdp_3 = output.shift(3)
gdp_sma3 = gdp_1.rolling(window = 3).mean()
gdp_sma6 = gdp_1.rolling(window=6).mean()
gdp_sma12 = gdp_1.rolling(window=12).mean()
gdp_ema6 = gdp_1.ewm(6).mean()
gdp_ema3 = gdp_1.ewm(3).mean()

#add GDP transforms to dataframe
baf2['GDP_1'] = gdp_1
baf2['GDP_2'] = gdp_2
baf2['GDP_3'] = gdp_3
baf2['GDP_SMA3'] = gdp_sma3
baf2['GDP_SMA6']  = gdp_sma6
baf2['GDP_SMA12'] = gdp_sma12
baf2['GDP_EMA6'] = gdp_ema6
baf2['GDP_EMA3'] = gdp_ema3

#add output to dataframe
baf2['GDP'] = output
baf2 = baf2.dropna()

'''
Create model
'''

X, y = baf2.values[:, 0:24], baf2.values[:, 24]
y = y.reshape(-1,1)
scalerx = pre.MinMaxScaler(feature_range=(0,1)).fit(X)
x_scale = scalerx.transform(X)
scalery = pre.MinMaxScaler(feature_range=(0,1)).fit(y)
y_scale = scalery.transform(y)

train_split = int(len(baf2)*0.75)

X_train, X_test = x_scale[0:train_split], x_scale[train_split:]
y_train, y_test = y_scale[0:train_split], y_scale[train_split:]

X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

#y_train = np.reshape(y_train, (y_train.shape[0], 1, 1))
#y_test = np.reshape(y_test, (y_test.shape[0], 1, 1))


#Create model
model = Sequential()
model.add(LSTM(64, input_shape = (1,24),activation='relu'))
model.add(Dense(8)) #activation='relu'#model.add(Dense(8))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train, y_train, epochs=200, batch_size=20, verbose=2)

#generate predictions for training
trainPredict = scalery.inverse_transform(model.predict(X_train))
testPredict = scalery.inverse_transform(model.predict(X_test))
totalPredict = np.concatenate((trainPredict,testPredict), axis=0)

trainScore = model.evaluate(X_train, y_train, verbose=0)
print ('train Score: %.2f MSE (%.2f RMSE)' % (trainScore, sqrt(trainScore)))
testScore = model.evaluate(X_test, y_test, verbose=0)
print ('Test Score: %.2f MSE (%.2f RMSE)' % (testScore, sqrt(testScore)))

# calculate root mean squared error
trainScore_2 = np.sqrt(mean_squared_error(baf2['GDP'], totalPredict))
print('Full Set Score: %.2f RMSE' % (trainScore_2))
#testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
#print('Test Score: %.2f RMSE' % (testScore))

#Create dataframe with existing GDP and also the predicted values
final_df = pd.DataFrame(baf2['GDP'])
final_df['pred'] = totalPredict
#create quick plot
final_df.plot()

#create Plotly plots
trace1 = go.Scatter(
                    x = final_df.index,
                    y = final_df['GDP'].values,
                    name = 'GDP YoY',
                    #color = '#4155f4'
#                    line = dict(
#                                color = ('#4155f4'),
#                                width = 1.5)
                    ) 

trace2 = go.Scatter(
                        x = final_df.index,
                        y = final_df['pred'].values,
                        name = 'Predicted GDP',
#                        yaxis = 'y2',
#                        line = dict(
#                                    color = ('#ccccff'),
#                                    width = 1.0,
#                                    ),
#                        fill = 'tonexty',
#                        opacity = 0.05,
                       
                           
    )       
        
layout  = {'title' : 'GDP Prediction_NN',
                   'xaxis' : {'title' : 'Date', 'type': 'date',
                              'fixedrange': True},
                   'yaxis' : {'title' : 'GDP YoY',
                              'fixedrange': True},
#                   'yaxis2' : {'title' : 'Assets',
#                               'overlaying' : 'y',
#                               'side'   : 'right'},
#                   'shapes': [{'type': 'rect',
#                              'x0': r[i]['scr_1y'].index[0],
#                              'y0': -2,
#                              'x1': r[i]['scr_1y'].index[-1],
#                              'y1': 2,
#                              'name': 'Z-range',
#                              'line': {
#                                      'color': '#f48641',
#                                      'width': 2,},
#                                      'fillcolor': '#f4ad42',
#                                      'opacity': 0.25,
#                                      },]
                   }
    
data = [trace1, trace2]
figure = go.Figure(data=data, layout=layout)
py.iplot(figure, filename = 'Macro_Data/GDP')


