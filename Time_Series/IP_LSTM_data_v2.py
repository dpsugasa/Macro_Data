# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 17:39:09 2018

@author: dsugasa
"""

import pandas as pd
from tia.bbg import LocalTerminal
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
import plotly
import plotly.plotly as py #for plotting
import plotly.offline as offline
import plotly.graph_objs as go
import plotly.tools as tls
#import cufflinks as cf
import plotly.figure_factory as ff
import credentials

#fix the random seed for reproducability (not sure what this is)
#np.random.seed(7)

#start script
start_time = datetime.now()

# set dates, securities, and fields

start_date = '05/04/1990'
end_date = "{:%m/%d/%Y}".format(datetime.now())
IDs = ['IP Index']
fields = ['LAST PRICE']
days = 'NON_TRADING_WEEKDAYS' #(NON_TRADING_WEEKDAYS | ALL_CALENDAR_DAYS | ACTIVE_DAYS_ONLY)
fill = 'PREVIOUS_VALUE' #(PREVIOUS_VALUE | NIL_VALUE)

df = LocalTerminal.get_historical(IDs, fields, start_date, end_date)
                                   #non_trading_day_fill_option = days,
                                   #non_trading_day_fill_method = fill)
df = df.as_frame()
df.columns = df.columns.droplevel(-1)

#make stationary
df_diff = df['IP Index'].diff(1).dropna().to_frame()

result = adfuller(df_diff.squeeze())
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))

#normalize the dataset
scaler = MinMaxScaler()
ds_new = scaler.fit_transform(df_diff)

df_scale = pd.DataFrame(ds_new, index=df_diff.index, columns = ['IP Index'])


df_scale['Shift_1'] = df_scale['IP Index'].shift(1)
df_scale['Shift_2'] = df_scale['IP Index'].shift(2)
df_scale['Shift_3'] = df_scale['IP Index'].shift(3)   #reshape into X=t and Y=t+1, t+2, t+3
df_scale['ewm_3'] = df_scale['IP Index'].ewm(3).mean()
df_scale['ewm_6'] = df_scale['IP Index'].ewm(6).mean()
df_scale['ewm_12'] = df_scale['IP Index'].ewm(12).mean()
df_scale['sma_3'] = df_scale['IP Index'].rolling(3).mean()
df_scale['sma_6'] = df_scale['IP Index'].rolling(6).mean()
df_scale['diff_3'] = df_scale['IP Index'].diff(3)
df_scale['diff_6'] = df_scale['IP Index'].diff(6)
df_scale['diff_12'] = df_scale['IP Index'].diff(12)
df_scale= df_scale.dropna()

dataset = df_scale.values
dataset = dataset.astype('float32')

#split into train and test sets
train_size = int(len(dataset)*0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

#reshape input to be [samples, time steps, features]
trainY = train[:,0]
trainX = train[:,1:12]
testY = test[:,0]
testX = test[:,1:12]

trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# create and fit the LSTM network
look_back = 11
model = Sequential()
model.add(LSTM(32, return_sequences=True, input_shape=(1, look_back)))
model.add(LSTM(32, return_sequences=True))
model.add(LSTM(32))
model.add(Dense(1, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model.fit(trainX, trainY, epochs=1000, batch_size=32, verbose=1)

# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

#inverse difference the original
df_fin = df['IP Index'].to_frame()
df_fin['IP_diff'] = df_diff['IP Index']
c = pd.Series(trainPredict.flatten(), index=df.index[10:len(trainPredict)+10])
c1 = pd.Series(c, index=df_fin.index)
df_fin['Predict_train'] = c1
d = pd.Series(testPredict.flatten(), index = df.index[len(trainPredict)+12:-1])
d1 = pd.Series(d, index=df_fin.index)
df_fin['Predict_test'] = d1
df_fin['IP_shift'] =  df['IP Index'].shift()
df_fin['pred_train_new'] = df_fin['Predict_train'] + df_fin['IP_shift']
df_fin['pred_test_new'] = df_fin['Predict_test'] + df_fin['IP_shift']

#df_fin[['IP Index', 'pred_train_new', 'pred_test_new']].iplot(filename='RFE/LSTM_IP_Predict')

# Create Probability Density by Strike
trace1 = go.Scatter(
            x = df_fin.index,
            y = df_fin['IP Index'],
            xaxis = 'x1',
            #yaxis = 'y1',
            mode = 'lines',
            line = dict(width=3, color= '#4d4dff'),
            name = 'IP Index',
            #text = format_prob,
            #textposition = 'top left',
            #textfont=dict(size=14),
            #marker = dict(size=12),
            #fill='tonexty',
            #fillcolor = '#b3b3b3'
            )
            
trace2 = go.Scatter(
            x = df_fin.index,
            y = df_fin['pred_train_new'],
            xaxis = 'x1',
            #yaxis = 'y1',
            mode = 'lines',
            line = dict(width=3, color= '#4dff4d'),
            name = 'train_new',
            #text = format_prob,
            #textposition = 'top left',
            #textfont=dict(size=14),
            #marker = dict(size=12),
            #fill='tonexty',
            #fillcolor = '#b3b3b3'
            )

trace3 = go.Scatter(
            x = df_fin.index,
            y = df_fin['pred_test_new'],
            xaxis = 'x1',
            #yaxis = 'y1',
            mode = 'lines',
            line = dict(width=3, color= '#ffff4d'),
            name = 'Default Probability',
            #text = format_prob,
            #textposition = 'top left',
            #textfont=dict(size=14),
            #marker = dict(size=12),
            #fill='tonexty',
            #fillcolor = '#b3b3b3'
            )

        
    
layout  = {'title' : 'IP Prediction',
                           'xaxis' : {'title' : 'Date',
                                      'fixedrange': True,
                                      'showgrid' : True},
                           'yaxis' : {'title' : 'IP',
                                      'fixedrange' : True,
                                      'showgrid' :True},
                            
        #                   'shapes': [{'type': 'rect',
        #                              'x0': d[i]['scr_1y'].index[0],
        #                              'y0': -2,
        #                              'x1': d[i]['scr_1y'].index[-1],
        #                              'y1': 2,
        #                              'name': 'Z-range',
        #                              'line': {
        #                                      'color': '#f48641',
        #                                      'width': 2,},
        #                                      'fillcolor': '#f4ad42',
        #                                      'opacity': 0.25,
        #                                      },]
                           }
data = [trace1, trace2, trace3]
figure = go.Figure(data = data, layout=layout)
py.iplot(figure, filename = 'Macro_Data/IP')

df_fin_test = df_fin[['IP Index', 'pred_test_new']]
df_fin_test = df_fin_test.dropna()
df_fin_train = df_fin[['IP Index', 'pred_train_new']]
df_fin_train = df_fin_train.dropna()

df_fin_test['IP_Index_YoY'] = df_fin_test['IP Index'].pct_change(12)*100
df_fin_test['IP_Pred_YoY'] = df_fin_test['pred_test_new'].pct_change(12)*100
df_fin_test = df_fin_test.dropna()
df_fin_train['IP_Index_YoY'] = df_fin_train['IP Index'].pct_change(12)*100
df_fin_train['IP_Pred_YoY'] = df_fin_train['pred_train_new'].pct_change(12)*100
df_fin_train = df_fin_train.dropna()

# calculate root mean squared error
trainScore = np.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = np.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))
full_trainScore = np.sqrt(mean_squared_error(df_fin_train['IP Index'], df_fin_train['pred_train_new']))
print('Full Train Score: %.2f RMSE' % (full_trainScore))
full_testScore = np.sqrt(mean_squared_error(df_fin_test['IP Index'], df_fin_test['pred_test_new']))
print('Full Test Score: %.2f RMSE' % (full_testScore))
full_testScore_yoy = np.sqrt(mean_squared_error(df_fin_test['IP_Index_YoY'], df_fin_test['IP_Pred_YoY']))
print('Full Test Score YoY: %.2f RMSE' % (full_testScore_yoy))
full_trainScore_yoy = np.sqrt(mean_squared_error(df_fin_train['IP_Index_YoY'], df_fin_train['IP_Pred_YoY']))
print('Full Train Score YoY: %.2f RMSE' % (full_trainScore_yoy))