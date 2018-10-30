# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 18:00:04 2018

@author: dsugasa
"""

import pandas as pd
import tia.bbg.datamgr as dm
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
#import sklearn
#from sklearn import linear_model
#from sklearn.model_selection import train_test_split
#from sklearn.svm import SVR
import fbprophet
from fbprophet.diagnostics import cross_validation

#set plot parameters
plt.rcParams['figure.figsize']= (100,60)
plt.style.use('seaborn')

#set script starting time
start_time = datetime.now()

# create a DataManager5 for simpler api access
mgr = dm.BbgDataManager()
# set dates, securities, and fields
start_date = '06/01/1950'
end_date = "{:%m/%d/%Y}".format(datetime.now())
IDs = ['CPI YOY Index']

sids = mgr[IDs]
fields = ['LAST PRICE']

df = sids.get_historical(fields, start_date, end_date)
df = df.fillna(method = 'ffill')
df.columns = df.columns.droplevel(-1)
#df = df.stack(level = 0, dropna=False)
#df['y_orig'] = df['CPI YOY Index']
#df['CPI YOY Index'] = np.log(df['CPI YOY Index'])

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


