# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 10:42:02 2018

Building a new GDP model. 
'Now Casting with Keras':  trying to predict YoY change in GDP
Changing the input data by using 12 month differencing

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
from sklearn.model_selection import TimeSeriesSplit
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import RFE
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
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
from fredapi import Fred
import lightgbm as lgb
from xgboost import XGBRegressor
from sklearn.metrics import accuracy_score
import credentials

fred = Fred(api_key=fred_api)

#set script starting time
start_time = datetime.now()

#def multi_diff(series, num=12):
#    for i in range(1,num):
#        series.shift(i)
#        return series


indics = {'INDPRO':   'IP',               #Industrial Production
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

non_stat = ['IP',
            'Exports',
            'Real_Exports',
            'NewOrders',
            'NewOrders_NoDef',
            'Real_PCE',
            'Real_Imports',
            'Imports',
            'Wk_hrs_manu',
            'Wk_hrs_priv',          
            'Tot_veh_sales',
            'Lt_wght_veh',
            'Heavy_trucks',
            ]

d = {} #dict of data

for code, name in indics.items():
    d[name] = fred.get_series_latest_release(code)
    d[name] = d[name].resample('M').last()
    d[name] = d[name].interpolate(method = 'linear')

for i in non_stat:
    d[i] =  d[i].diff(12)
    
'''
#######################
Create the final prediction dataframe and array of predictions
#######################
'''

#final dataframe with new predictions; will be fed to the GDP model
pred_periods = 6
fin_idx = d['UMich'].index.union(pd.date_range(d['UMich'].index[-1]+1,
                                   periods=pred_periods,
                                   freq=d['UMich'].index.freq))
pred_df = pd.DataFrame(index=fin_idx)

#array to append predictions to


    
d['UMich'] = pd.DataFrame(d['UMich'], columns = ['Last'])
b_snax = d['UMich']['Last'].values
    
# create multiple features with shifted dated
for i in range(1,13):
    d['UMich'][f'Shift_{i}'] = d['UMich']['Last'].shift(i)
    
#create multiple features with diff data

for i in range(1,13):
    d['UMich'][f'Diff_{i}'] = d['UMich']['Shift_1'].diff(i)
    
#create some moving average features
for i in range(3,10):
    d['UMich'][f'SMA_{i}'] = d['UMich']['Shift_1'].rolling(window=i).mean()
    d['UMich'][f'EMA_{i}'] = d['UMich']['Shift_1'].ewm(i).mean()
    
d['UMich'] = d['UMich'].fillna(method='ffill').dropna()

#create a dataframe that will be used for new predictions
pred_df = pd.DataFrame(d['UMich']['Last'])
pred_df = pred_df.fillna(method = 'ffill')
for i in range(1,12):
    pred_df[f'Shift_{i}'] = pred_df['Last'].shift(i)
    
for i in range(1,13):
    pred_df[f'Diff_{i}'] = pred_df['Last'].diff(i)
    
#create some moving average features
for i in range(3,10):
    pred_df[f'SMA_{i}'] = pred_df['Last'].rolling(window=i).mean()
    pred_df[f'EMA_{i}'] = pred_df['Last'].ewm(i).mean()
    
pred_niner = pred_df.iloc[-1].values
pred_niner = pred_niner.reshape(1,-1)
#pred_niner = np.reshape(pred_niner, (pred_niner.shape[0], 1, pred_niner.shape[1]))
    

X, y = d['UMich'].values[:, 1:39], d['UMich']['Last'].values
y = y.reshape(-1,1)
scalerx = pre.MinMaxScaler(feature_range=(0,1)).fit(X)
x_scale = scalerx.transform(X)
scalery = pre.MinMaxScaler(feature_range=(0,1)).fit(y)
y_scale = scalery.transform(y)

train_split = int(len(d['UMich'])*0.75)

X_train, X_test = x_scale[0:train_split], x_scale[train_split:]
y_train, y_test = y_scale[0:train_split], y_scale[train_split:]

X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

#Create model
model = Sequential()
model.add(LSTM(32, input_shape = (1,38)) )#activation='relu'))
model.add(Dense(16)) #activation='relu'#model.add(Dense(8))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train, y_train, epochs=1000, batch_size=20, verbose=2)

#generate predictions for training
trainPredict = scalery.inverse_transform(model.predict(X_train))
testPredict = scalery.inverse_transform(model.predict(X_test))
totalPredict = np.concatenate((trainPredict,testPredict), axis=0)

trainScore = model.evaluate(X_train, y_train, verbose=0)
print ('LSTM_RMSE_Train Score: %.4f' % (sqrt(trainScore)))
testScore = model.evaluate(X_test, y_test, verbose=0)
print ('LSTM_RMSE_Test Score: %.4f' % (sqrt(testScore)))

# calculate root mean squared error
trainScore_2 = np.sqrt(mean_squared_error(d['UMich']['Last'], totalPredict))
print('LSTM_RMSE_Full  Score: %.4f RMSE' % (trainScore_2))

#Create dataframe with existing GDP and also the predicted values
final_df = pd.DataFrame(d['UMich']['Last'])
final_df['pred'] = totalPredict

#create quick plot
final_df.plot()

new_pred = scalerx.transform(pred_niner)
new_pred = np.reshape(new_pred, (new_pred.shape[0], 1, new_pred.shape[1]))
next_2 = scalery.inverse_transform(model.predict((new_pred)))
print(next_2)


fuck = final_df.index.union(pd.date_range(final_df.index[-1]+1, periods=2, freq=final_df.index.freq))
 
dicky = pd.DataFrame(index = fuck)   
    
    
    
    
'''   
frames = [d[i] for i in indics.values()]
columns = [i for i in indics.values()]

baf = pd.concat(frames, keys = columns, join = 'outer', axis = 1)
baf = baf.fillna(method = 'ffill')


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


'''
#Create Ouput

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
gdp_diff3 = gdp_1.diff(3)
gdp_diff6 = gdp_1.diff(6)
gdp_diff9 = gdp_1.diff(9)
gdp_diff12 = gdp_1.diff(12)
gdp_diff2 = gdp_1.diff(2)

#add GDP transforms to dataframe
baf2['GDP_1'] = gdp_1
baf2['GDP_2'] = gdp_2
baf2['GDP_3'] = gdp_3
baf2['GDP_SMA3'] = gdp_sma3
baf2['GDP_SMA6']  = gdp_sma6
baf2['GDP_SMA12'] = gdp_sma12
baf2['GDP_EMA6'] = gdp_ema6
baf2['GDP_EMA3'] = gdp_ema3
baf2['GDP_diff3'] = gdp_diff3
baf2['GDP_diff6'] = gdp_diff6
baf2['GDP_diff9'] = gdp_diff9
baf2['GDP_diff12'] = gdp_diff12
baf2['GDP_diff2'] = gdp_diff2

#create a dataframe that will be used for new predictions
pred_df = baf2
pred_df = pred_df.fillna(method = 'ffill')
niner = pred_df.iloc[-1].values
niner = niner.reshape(1,-1)


#add output to dataframe
baf2['GDP'] = output
baf2 = baf2.dropna()

'''
#Create model
'''

X, y = baf2.values[:, 0:29], baf2.values[:, 29]
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
model.add(LSTM(64, input_shape = (1,29),activation='relu'))
model.add(Dense(8)) #activation='relu'#model.add(Dense(8))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train, y_train, epochs=600, batch_size=20, verbose=2)

#generate predictions for training
trainPredict = scalery.inverse_transform(model.predict(X_train))
testPredict = scalery.inverse_transform(model.predict(X_test))
totalPredict = np.concatenate((trainPredict,testPredict), axis=0)

trainScore = model.evaluate(X_train, y_train, verbose=0)
print ('LSTM_RMSE_Train Score: %.4f' % (sqrt(trainScore)))
testScore = model.evaluate(X_test, y_test, verbose=0)
print ('LSTM_RMSE_Test Score: %.4f' % (sqrt(testScore)))

# calculate root mean squared error
trainScore_2 = np.sqrt(mean_squared_error(baf2['GDP'], totalPredict))
print('LSTM_RMSE_Full  Score: %.4f RMSE' % (trainScore_2))

#Create dataframe with existing GDP and also the predicted values
final_df = pd.DataFrame(baf2['GDP'])
final_df['pred'] = totalPredict

#create quick plot
final_df.plot()

#create Plotly plots
trace1 = go.Scatter(
                    x = final_df.index,
                    y = final_df['GDP'].values,
                    name = 'GDP % YoY',
                    line = dict(
                                color = ('#0000cc'),
                                width = 1.5)
                    ) 

trace2 = go.Scatter(
                        x = final_df.index,
                        y = final_df['pred'].values,
                        name = 'Predicted GDP',
                        line = dict(
                                    color = ('#ffa500'),
                                    width = 1.5,
                                    ),

    )       
        
layout  = {'title' : 'GDP Prediction_LSTM',
                   'xaxis' : {'title' : 'Date', 'type': 'date',
                              'fixedrange': True},
                   'yaxis' : {'title' : 'GDP % YoY',
                              'fixedrange': True},

                   }
    
data = [trace1, trace2]
figure = go.Figure(data=data, layout=layout)
py.iplot(figure, filename = 'Macro_Data/GDP/LSTM/full-series')


'''
#####################
#now try Random Forest
#####################
'''

X, y = baf2.values[:, 0:29], baf2.values[:, 29]

train_split = int(len(baf2)*0.75)

X_train, X_test = X[0:train_split], X[train_split:]
y_train, y_test = y[0:train_split], y[train_split:]

#find best parameters
def Grid_Search_CV_RFR(X_train, y_train):
    reg = RandomForestRegressor()
    param_grid = { 
            "n_estimators"      : [10,25,50,100,500],
            "max_features"      : ["auto"],
            "min_samples_leaf" : [1,5,10,25,50,100]
            }

    tss_splits = TimeSeriesSplit(n_splits=10).split(X_train)
    grid = GridSearchCV(reg, param_grid, cv=tss_splits, verbose=1) 

    grid.fit(X_train, y_train)

    return grid.best_score_ , grid.best_params_

best_score, best_params = Grid_Search_CV_RFR(X_train, y_train)

mf = best_params['max_features']
msl = best_params['min_samples_leaf']
ne = best_params['n_estimators']

rfr = RandomForestRegressor(n_estimators=ne, max_features=mf, min_samples_leaf=msl, random_state=1)
rfr.fit(X_train, y_train)

features = baf2.iloc[:,0:29].columns
importances = rfr.feature_importances_
indices = np.argsort(importances)

trace = go.Bar(
    x=features[indices],
    y=importances[indices],
    marker = dict(color='green')
)

data=[trace]

# Edit the layout, then plot!
layout = dict(title = 'GDP Feature Importance (RF)',
              yaxis = dict(title = 'Relative Importance',
                           showgrid = True,
                           fixedrange = True),
              xaxis = dict(autorange='reversed',
                           tickfont=dict(size=10),
                           showgrid = True,
                           fixedrange = True)
              )

fig = dict(data=data, layout=layout)
py.iplot(fig, filename='Macro_Data/GDP/RFE/feature_importance')

#predict values
train_pred = rfr.predict(X_train)
test_pred = rfr.predict(X_test)
tot_pred = np.append(train_pred, test_pred)

final_df2 = pd.DataFrame(baf2['GDP'])
final_df2['pred'] = tot_pred

rmse_train = np.sqrt(mean_squared_error(y_train, train_pred))
rmse_test = np.sqrt(mean_squared_error(y_test, test_pred))
rmse_full = np.sqrt(mean_squared_error(final_df2['GDP'], final_df2['pred']))

print('RF_RMSE_train: %.4f' % rmse_train)
print('RF_RMSE_full: %.4f' % rmse_full)


# Create traces.
def create_trace(df, color, label):
    dates = df.index 
    prices = df.values

    trace = go.Scatter(
        x = dates,
        y = prices,
        name = label,
        line = dict(color=color,
                    width=1.5)
    )
    return trace


#plot model performance
pred_trace = create_trace(final_df2['pred'], '#ffa500', 'Predicted_GDP')
act_trace = create_trace(final_df2['GDP'], '#0000cc', 'GDP % YoY')
data = [pred_trace, act_trace]

# Edit the layout, then plot!
layout = dict(title = 'GDP Prediction_RF',
              xaxis = dict(title = 'Date',
                           fixedrange=True),
              yaxis = dict(title = 'Last',
                           fixedrange=True),
              )

fig = dict(data=data, layout=layout)

py.iplot(fig, filename='Macro_Data/GDP/RFE/full-series')

next_1 = rfr.predict(niner)
niner_scale = scalerx.transform(niner)
niner_scale = np.reshape(niner_scale, (niner_scale.shape[0], 1, niner_scale.shape[1]))
next_2 = scalery.inverse_transform(model.predict(niner_scale))

print('RF_next_GDP %.3f' % next_1)
print('LSTM_next_GDP %.3f' % next_2)

'''
#####################
#now try LightBGM
#####################
'''

X, y = baf2.values[:, 0:29], baf2.values[:, 29]

train_split = int(len(baf2)*0.75)

X_train, X_test = X[0:train_split], X[train_split:]
y_train, y_test = y[0:train_split], y[train_split:]

# create dataset for lightgbm
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

# specify your configurations as a dict
params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': {'l2', 'l1'},
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}

print('Starting training...')

evals_result = {}
feature_name = list(baf2.columns[0:29])

# train
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=100,
                valid_sets=lgb_eval,
                feature_name = feature_name,
                #categorical_feature=[21],
                evals_result=evals_result,
                verbose_eval=0)
#               early_stopping_rounds=50)



print('Saving model...')
# save model to file
gbm.save_model('model.txt')

print('Starting predicting...')
# predict
#y_pred_train = gbm.predict(X_train, num_iteration=gbm.best_iteration)
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
# eval
print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)

fin_df = pd.DataFrame(y_pred, columns = ['y_pred'], index = baf2.index[train_split:])
fin_df['actual'] = baf2['GDP'][train_split:]

def render_plot_importance(importance_type, max_features=10,
                           ignore_zero=True, precision=4):
    ax = lgb.plot_importance(gbm, importance_type=importance_type,
                             max_num_features=max_features,
                             ignore_zero=ignore_zero, figsize=(12, 8),
                             precision=precision)
    plt.show()
    
render_plot_importance(importance_type='split')

'''
#####################
#now try XGBoost
#####################
'''
X, y = baf2.values[:, 0:29], baf2.values[:, 29]

train_split = int(len(baf2)*0.75)

X_train, X_test = X[0:train_split], X[train_split:]
y_train, y_test = y[0:train_split], y[train_split:]

model = XGBRegressor()
model.fit(X_train, y_train)
# make predictions for test data
y_pred = model.predict(X_test)
#predictions = [round(value) for value in y_pred]
# evaluate predictions
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred))
print('XGB_RMSE_test: %.4f' % rmse_test)


'''
print ("Time to complete:", datetime.now() - start_time)






