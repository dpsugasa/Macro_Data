# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 11:22:56 2018

@author: dsugasa
"""


import pandas as pd
import tia.bbg.datamgr as dm
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
#import sklearn
#from sklearn import linear_model
#from sklearn.model_selection import train_test_split
#from sklearn.svm import SVR
import fbprophet
from fbprophet.diagnostics import cross_validation
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

#set plot parameters
plt.rcParams['figure.figsize']= (20,10)
plt.style.use('seaborn')

#set the script start time
start_time = datetime.now()

def mape_vectorized_v2(a, b): 
    mask = a != 0.0
    return (np.fabs(a - b)/a)[mask].mean() 


# create a DataManager5 for simpler api access
mgr = dm.BbgDataManager()
# set dates, securities, and fields
start_date = "{:%m/%d/%Y}".format(datetime.now() - relativedelta(days=500)) #500 trading days
end_date = "{:%m/%d/%Y}".format(datetime.now())
IDs = ['SCR FP Equity', 'SX5E Index', 'UCG IM Equity']

sids = mgr[IDs]
fields = ['LAST PRICE', 'HIGH', 'LOW']

####Create STARC Bands###

periods=5
sds=1.05
ATR_Period=15



d = {} #dict of original dataframes per ID
m = {} #dict of models
z = {} #dict of dataframes with appended forecasts
k = {} #dict of forecast dataframes transformed back to original scale
ev = {} #dict of metrics
y = {} #dict of dataframes used for cross-validation


for name in IDs:
    #build original dataframes with log transform
    d[name] = mgr[name].get_historical(fields, start_date, end_date)
    d[name] = d[name].fillna(method='ffill')
    #d[name] = d[name])
    d[name] = d[name].rename(columns={'LAST PRICE': 'y'})
    d[name]['y'] = d[name]['y']
    d[name]['ds'] = d[name].index
    d[name] = d[name].fillna(method = 'ffill')
    
    
    d[name]['Prev Close'] = d[name]['y'].shift(1) #simplify prev close
    d[name]['ActR'] = abs(d[name]['HIGH']-d[name]['LOW']) #define actual trading range
    d[name]['TRHigh'] = abs(d[name]['HIGH']-d[name]['Prev Close']) #absolute value of diff prev C to H
    d[name]['TRLow'] = abs(d[name]['LOW']-d[name]['Prev Close']) #absolute value of diff prev C to L
    d[name]['True Range'] = d[name][['ActR','TRHigh','TRLow']].apply(max,axis=1) #define true range
    
    d[name]['ATR_E'] = d[name]['True Range'].ewm(span=ATR_Period).mean() #exponential ATR
    d[name]['MA_E'] = d[name]['y'].ewm(span=periods).mean()
    d[name]['UB']=d[name]['MA_E']+(d[name]['ATR_E']*sds)
    d[name]['LB']=d[name]['MA_E']-(d[name]['ATR_E']*sds)
    d[name]['STARCWidth']=d[name]['UB']-d[name]['LB']
    d[name]['STARC%'] = ((d[name]['y'] - d[name]['LB'])/d[name]['STARCWidth'])*100
    
    #build models and fit to data
    m[name] = fbprophet.Prophet(changepoint_prior_scale=0.15,
                 weekly_seasonality=False, yearly_seasonality=True,
                 interval_width=0.8)
    m[name].fit(d[name])
    
    #create new dataframes for predictions; still in transform scale
    z[name] = m[name].make_future_dataframe(periods=15, freq='D')
    z[name] = m[name].predict(z[name])
    
    #move back to original scale
    k[name] = z[name][['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    #k[name].loc[:,'yhat'] = np.exp(k[name]['yhat'])
    #k[name].loc[:, 'yhat_lower'] = np.exp(k[name]['yhat_lower'])
    #k[name].loc[:, 'yhat_upper'] = np.exp(k[name]['yhat_upper'])
#    k[name]['UB'] = d[name]['UB']
#    k[name]['LB'] = d[name]['LB']
#    k[name]['MA'] = d[name]['MA_E']
    k[name] = k[name].set_index('ds')
    
    #plot forecasts in log scale
    m[name].plot(z[name], xlabel = 'Date', ylabel = name)
    m[name].plot_components(z[name])
    
    #add original data to forecast data to analyze accuracy
    ev[name] = pd.concat([k[name][['yhat','yhat_upper', 'yhat_lower']],
                         d[name][['y','UB','LB','MA_E']]], join='outer', axis=1)
    ev[name] = ev[name].dropna()
    #m[name].plot(ev[name], xlabel = 'Date', ylabel = name)
    
    print(name, "R2: ", r2_score(ev[name]['y'], ev[name]['yhat']))
    print(name, "R2_MA_E: ", r2_score(ev[name]['y'], ev[name]['MA_E']))
    print(name, "RMSE: ", np.sqrt(mean_squared_error(ev[name]['y'], ev[name]['yhat'])))
    print(name, "MAE: ", mean_absolute_error(ev[name]['y'], ev[name]['yhat']))
    print(name, "MAPE: ", mape_vectorized_v2(ev[name]['y'], ev[name]['yhat']))
    print(name, "RMSE_MA_E: ", np.sqrt(mean_squared_error(ev[name]['y'], ev[name]['MA_E'])))
    print(name, "RMSE_UB: ", np.sqrt(mean_squared_error(ev[name]['yhat_upper'], ev[name]['UB'])))
    print(name, "RMSE_LB: ", np.sqrt(mean_squared_error(ev[name]['yhat_lower'], ev[name]['LB'])))
    
    
            
    #cross validation takes forever
    y[name] = cross_validation(m[name], horizon = '90 days')
    #y[name]['y_act'] = np.exp(y[name]['y'])
    #y[name]['y_predict'] = np.exp(y[name]['yhat'])
    #print(df_cv.tail())
    #print(name, "RMSE_CrossVal: ", np.sqrt(mean_squared_error(y[name]['y'], y[name]['yhat'])))
    ev[name][['yhat_upper', 'UB']].plot()
    ev[name][['yhat', 'MA_E']].plot()
    ev[name]['HIGH'] = d[name]['HIGH']
    ev[name]['LOW'] = d[name]['LOW']
    #ev[name]['y'] = np.exp(d[name]['y'])
    ev[name]['ST_Miss'] = ev[name].apply(lambda x: 1 if x['HIGH'] > x['UB'] or x['LOW'] < x['LB'] else 0, axis=1)
    ev[name]['ST_Hit'] = ev[name].apply(lambda x: 1 if x['y'] < x['UB'] and x['y'] > x['LB'] else 0, axis=1)
    ev[name]['yhat_Miss'] = ev[name].apply(lambda x: 1 if x['HIGH'] > x['yhat_upper'] or x['LOW'] < x['yhat_lower'] else 0, axis=1)
    ev[name]['yhat_Hit'] = ev[name].apply(lambda x: 1 if x['y'] < x['yhat_upper'] and x['y'] > x['yhat_lower'] else 0, axis=1)
    print ("ST Miss: %s"  % (ev[name]['ST_Miss'].sum()/len(ev[name]['y'])))
    print ("ST Hit: %s"  % (ev[name]['ST_Hit'].sum()/len(ev[name]['y'])))
    print ("yhat Miss: %s"  % (ev[name]['yhat_Miss'].sum()/len(ev[name]['y'])))
    print ("yhat Hit: %s"  % (ev[name]['yhat_Hit'].sum()/len(ev[name]['y'])))
    
#measure the time it takes to run the script

#start_time = datetime.now()
print ("Time to complete:", datetime.now() - start_time)




####Create Multi-Period Moving Averages####

#ma_30 = df['LAST PRICE'].unstack().ewm(span=30).mean()
#df['MA_30'] = ma_30.stack()
#
#ma_50 = df['LAST PRICE'].unstack().ewm(span=50).mean()
#df['MA_50'] = ma_50.stack()
#
#ma_200 = df['LAST PRICE'].unstack().ewm(span=200).mean()
#df['MA_200'] = ma_200.stack()
#
#df['Signal30'] = df.apply(lambda x: "Long" if x['LAST PRICE'] > x['MA_30'] else "Short", axis=1)
#df['Signal50'] = df.apply(lambda x: "Long" if x['LAST PRICE'] > x['MA_50'] else "Short", axis=1)
#df['Signal200'] = df.apply(lambda x: "Long" if x['LAST PRICE'] > x['MA_200'] else "Short", axis=1)
#
#df['%30'] = (df['LAST PRICE'] - df['MA_30'])/df['LAST PRICE']*100
#df['%50'] = (df['LAST PRICE'] - df['MA_50'])/df['LAST PRICE']*100
#df['%200'] = (df['LAST PRICE'] - df['MA_200'])/df['LAST PRICE']*100
 

#ev[name]['ST_Miss'] = df.apply(lambda x: 1 if x['HIGH'] > x['UB'] or x['LOW'] < x['LB'] else 0, axis=1)
##df['ST_Miss'] = Miss.stack()
#df['ST_Hit'] = df.apply(lambda x: 1 if x['LAST PRICE'] < x['UB'] and x['LAST PRICE'] > x['LB'] else 0, axis = 1)
##df['ST_Hit'] = Hit.stack()
#st_hit = df['ST_Hit'].unstack().sum()
#st_miss = df['ST_Miss'].unstack().sum()
#t_range = len(df['STARCWidth'].unstack())
#
#starc_misses = st_miss/t_range
#starc_hits = st_hit/t_range

#print "ST Miss: %s"  % starc_misses
#print "ST Hit: %s"  % starc_hits
#idx = pd.IndexSlice
#print (df.loc[idx[:,['LAST PRICE','LB','UB','STARC%','Signal30','Signal50','Signal200',
#                    '%50', '%200']]].tail(len(IDs)).sort_values(by = 'STARC%'))