# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 22:27:45 2018

@author: dpsugasa
"""


import pandas as pd
import tia.bbg.datamgr as dm
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import quandl
#import sklearn
#from sklearn import linear_model
#from sklearn.model_selection import train_test_split
#from sklearn.svm import SVR
import fbprophet
from fbprophet.diagnostics import cross_validation, performance_metrics
#from fredapi import Fred
import credentials

#fred = Fred(api_key=fred_api)

#set script starting time
start_time = datetime.now()


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
df_fcast = {}
dfp = {}
cv = {}
perf = {}

for code, name in indics.items():
    d[name] = fred.get_series_latest_release(code)
    d[name] = d[name].resample('B').last()
    d[name] = d[name].interpolate(method = 'linear')
    d[name] = d[name].to_frame()
    
#    d[name] = d[name].rename(columns={0 : 'y'})
#    d[name]['y'] = d[name]['y']
#    d[name]['ds'] = d[name].index
#    dfp[name] = fbprophet.Prophet(
#                 weekly_seasonality=False, yearly_seasonality=True,
#                 ).fit(d[name])
#    df_fcast[name] = dfp[name].make_future_dataframe(periods=6, freq = 'M')
#    df_fcast[name] = dfp[name].predict(df_fcast[name])
#    cv[name] = cross_validation(dfp[name], horizon = '12 days')
#    perf[name] = performance_metrics(cv[name])


d['IP'] = d['IP'].rename(columns = {0:'y'})
d['IP']['y'] = d['IP']['y']
d['IP']['ds'] = d['IP'].index

dfp['IP'] = fbprophet.Prophet(seasonality_mode = 'multiplicative').fit(d['IP'])
df_fcast['IP'] = dfp['IP'].make_future_dataframe(periods = 180, freq = 'D')
df_fcast['IP'] = dfp['IP'].predict(df_fcast['IP'])

cv['IP'] = cross_validation(dfp['IP'], horizon = '180 days' )





# Make a future dataframe for 2 years
df_forecast = df_prophet.make_future_dataframe(periods=3, freq = 'M')
# Make predictions
df_forecast = df_prophet.predict(df_forecast)
    

#for i in non_stat:
#    d[i] =  d[i].diff(12)
   
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

#set plot parameters
plt.rcParams['figure.figsize']= (100,60)
plt.style.use('seaborn')



df = df.rename(columns={'date': 'ds', 'CPI YOY Index': 'y'})
df['ds'] = df.index

df_prophet = fbprophet.Prophet(changepoint_prior_scale=0.75,
                 weekly_seasonality=False, yearly_seasonality=False,
                 ).fit(df)
#df_prophet.add_regressor(df['VOLUME'])
#df_prophet.fit(df)

# Make a future dataframe for 2 years
df_forecast = df_prophet.make_future_dataframe(periods=3, freq = 'M')
# Make predictions
df_forecast = df_prophet.predict(df_forecast)

fig = df_prophet.plot(df_forecast, xlabel = 'Date', ylabel = 'CPI')
plt.title('CPI Price Action')
#for cp in df_prophet.changepoints:
#    plt.axvline(cp, c='gray', ls = '--', lw=2)

#df_cv = cross_validation(df_prophet, horizon = '30 days')
#print(df_cv.tail())

df_prophet.plot_components(df_forecast)

deltas = df_prophet.params['delta'].mean(0)
fig = plt.figure(facecolor = 'w', figsize=(10,6))
ax = fig.add_subplot(111)
ax.bar(range(len(deltas)), deltas, facecolor='#0072B2', edgecolor = '#0072B2')
ax.grid(True, which='major', c='gray', ls = '-', lw=1, alpha=0.2) 
ax.set_ylabel('Rate change')
ax.set_xlabel('Potential changepoint')
fig.tight_layout()

#measure time
print ("Time to complete:", datetime.now() - start_time)
