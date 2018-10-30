# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 14:18:32 2018

@author: dpsugasa
"""

#import all modules
import pandas as pd
import tia.bbg.datamgr as dm
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.stats import boxcox
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_log_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import RFE
#from lightgbm import LGBMRegressor
import plotly
import plotly.plotly as py #for plotting
import plotly.offline as offline
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.figure_factory as ff
import cufflinks as cf
import credentials
import quandl

np.random.seed(123)

#set the script start time
start_time = datetime.now()

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
df_total_2 = df_total
df_total = df_total.drop(['GDP','Retail_Sales', 'IP','SPX'], axis=1)


X = df_total.iloc[:,0:8].values
y = df_total.iloc[:,8:].values.ravel()

split_point = 0.8
train_size = int(len(X)*split_point)
test_size = len(X) - train_size
trainX, testX = X[0:train_size,:], X[train_size:len(X), :]
trainY, testY = y[0:train_size,], y[train_size:len(X), ]

#find best parameters
def Grid_Search_CV_RFR(X_train, y_train):
    reg = RandomForestRegressor()
    param_grid = { 
            "n_estimators"      : [100,500,1000],
            "max_features"      : ["auto", 4, 6],
            "min_samples_leaf" : [1,5,10,20]
            }

    tss_splits = TimeSeriesSplit(n_splits=10).split(X_train)
    grid = GridSearchCV(reg, param_grid, cv=tss_splits, verbose=0) #, cv=tss_splits
    #grid = GridSearchCV(reg, param_grid, cv=3, verbose=0)

    grid.fit(X_train, y_train)

    return grid.best_score_ , grid.best_params_

best_score, best_params = Grid_Search_CV_RFR(trainX, trainY)

mf = best_params['max_features']
msl = best_params['min_samples_leaf']
ne = best_params['n_estimators']

rfr = RandomForestRegressor(n_estimators=ne, max_features=mf, min_samples_leaf=msl, random_state=1)
rfr.fit(trainX, trainY)

features = df_total.iloc[:,0:8].columns
importances = rfr.feature_importances_
indices = np.argsort(importances)

trace = go.Bar(
    x=features[indices],
    y=importances[indices],
    marker = dict(color='green')
)

data=[trace]

# Edit the layout, then plot!
layout = dict(title = 'Feature Importance (RF)',
              yaxis = dict(title = 'Relative Importance',
                           showgrid = True,
                           fixedrange = True),
              xaxis = dict(autorange='reversed',
                           tickfont=dict(size=10),
                           showgrid = True,
                           fixedrange = True)
              )

fig = dict(data=data, layout=layout)
py.iplot(fig, filename='RFE/feature_importance')

#predict values
train_pred = rfr.predict(trainX)
test_pred = rfr.predict(testX)

#inverse difference
Tr_data = pd.DataFrame(trainY, columns = ['GDP_Shift'], index = df_total.index[:len(trainY)])
Tr_data['GDP'] = df_total_2['GDP']
Tr_data['Train_pred'] = train_pred
Tr_data['GDP_pred']  = Tr_data['GDP'].shift(1) + Tr_data['Train_pred']
Tr_data = Tr_data.dropna()

Te_data = pd.DataFrame(testY, columns = ['GDP_Shift'], index = df_total.index[len(trainY):])
Te_data['GDP'] = df_total_2['GDP']
Te_data['Test_pred'] = test_pred
Te_data['GDP_pred']  = Te_data['GDP'].shift(1) + Te_data['Test_pred']
Te_data = Te_data.dropna()

rmse_inv_test = np.sqrt(mean_squared_error(Te_data['GDP'].values, Te_data['GDP_pred'].values))
rmse_inv_train = np.sqrt(mean_squared_error(Tr_data['GDP'].values, Tr_data['GDP_pred'].values))
rmse_train = np.sqrt(mean_squared_error(trainY, train_pred))
rmse_test = np.sqrt(mean_squared_error(testY, test_pred))

print('RMSE_inv_test: %.3f' % rmse_inv_test)
print('RMSE_inv_train: %.3f' % rmse_inv_train)
print('RMSE_train: %.3f' % rmse_train)
print('RMSE_test: %.3f' % rmse_test)

# Create traces.
def create_trace(df, color, label):
    dates = df.index 
    prices = df.values

    trace = go.Scatter(
        x = dates,
        y = prices,
        name = label,
        line = dict(color=color)
    )
    return trace


#plot training performance
train_act_df = pd.DataFrame(trainY).set_index(df_total.index[:len(trainX)])
train_act_df.columns = ['act_close']
train_pred_df = pd.DataFrame(train_pred).set_index(df_total.index[:len(trainX)])
train_pred_df.columns = ['pred_close']

pred_trace = create_trace(train_pred_df, 'red', 'Predicted')
act_trace = create_trace(train_act_df, 'blue', 'Actual')
data = [pred_trace, act_trace]

# Edit the layout, then plot!
layout = dict(title = 'IP (Training)',
              xaxis = dict(title = 'Date',
                           fixedrange=True),
              yaxis = dict(title = 'Last',
                           fixedrange=True),
              )

fig = dict(data=data, layout=layout)
py.iplot(fig, filename='RFE/training-prices')

#plot test performance
test_act_df = pd.DataFrame(testY).set_index(df_total.index[len(trainX):])
test_act_df.columns = ['act_close']
test_pred_df = pd.DataFrame(test_pred).set_index(df_total.index[len(trainX):])
test_pred_df.columns = ['pred_close']

pred_trace = create_trace(test_pred_df, 'red', 'Predicted')
act_trace = create_trace(test_act_df, 'blue', 'Actual')
data = [pred_trace, act_trace]

# Edit the layout, then plot!
layout = dict(title = 'IP (Test)',
              xaxis = dict(title = 'Date'),
              yaxis = dict(title = 'Closing Price'),
              )

fig = dict(data=data, layout=layout)
py.iplot(fig, filename='RFE/test-prices')

frames = [train_act_df, test_act_df]
big_df  = pd.concat(frames, join = 'outer', axis=0)

frames_2 = [train_pred_df, test_pred_df]
big_df_2  = pd.concat(frames_2, join = 'outer', axis=0)

frames_3 = [big_df, big_df_2]
final_df = pd.concat(frames_3, join = 'outer', axis=1)

full_act_trace = create_trace(final_df['act_close'], 'green', 'Actual')
full_pred_trace = create_trace(final_df['pred_close'], 'blue', 'Predicted')

data = [full_pred_trace, full_act_trace]

# Edit the layout, then plot!
layout = dict(title = 'IP (Training)',
              xaxis = dict(title = 'Date',
                           fixedrange=True),
              yaxis = dict(title = 'Last',
                           fixedrange=True),
              )

fig = dict(data=data, layout=layout)
py.iplot(fig, filename='RFE/full-series')




print ("Time to complete:", datetime.now() - start_time)


