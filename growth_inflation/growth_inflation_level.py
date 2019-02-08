# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 23:22:35 2019

@author: dpsugasa
"""

#import all modules
import pandas as pd
from pandas.tseries.offsets import *
import numpy as np
import matplotlib.pyplot as plt
import datetime #for dates
from datetime import datetime
import quandl #for data
from math import sqrt
from tia.bbg import LocalTerminal
#import plotly
import plotly.plotly as py #for plotting
import plotly.offline as offline
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.figure_factory as ff
from fredapi import Fred
import credentials

fred = credentials.fred

#set the script start time
start_time = datetime.now()
date_now =  "{:%m_%d_%Y}".format(datetime.now())

start_date = '01/01/1950'
end_date = "{:%m/%d/%Y}".format(datetime.now())
IDs = ['GDP CYOY Index', 'CPI YOY Index'] 
fields = ['LAST PRICE']

df = LocalTerminal.get_historical(IDs, fields, start_date, end_date).as_frame() #period = 'QUARTERLY',
                                         #non_trading_day_fill_option = 'ALL_CALENDAR_DAYS',
                                         #non_trading_day_fill_method = 'PREVIOUS_VALUE').as_frame()
df.columns = df.columns.droplevel(-1)
df = df.resample('Q').mean()
df = df.dropna()

df['gdp_ror'] = df['GDP CYOY Index'].pct_change()
df['cpi_ror'] = df['CPI YOY Index'].pct_change()


df['gdp_dir'] = df.apply(lambda x: 1 if x['gdp_ror'] > 0 else(-1 if \
                              x['gdp_ror'] < 0 else 0), axis = 1)
df['gdp_dir'] = df['gdp_dir'].replace(to_replace = 0, method = 'ffill')

df['cpi_dir'] = df.apply(lambda x: 1 if x['cpi_ror'] > 0 else(-1 if \
                              x['cpi_ror'] < 0 else 0), axis = 1)
df['cpi_dir'] = df['cpi_dir'].replace(to_replace = 0, method = 'ffill')

#df['consec_gro'] = df['gdp_dir'] * (df['gdp_dir'].groupby((df['gdp_dir'] != df['gdp_dir'].shift()).cumsum()).cumcount()+1)
#df['dir_change'] = df['gdp_dir'] != df['gdp_dir'].shift()
#df['vol_watch'] = np.where(df['consec_gro'].shift() > 2 , 1,0)
#df['danger'] = df.apply(lambda x: 1 if x['vol_watch'] ==1 and x['dir_change'] == True else 0, axis =1)
#df['vol_watch'] = df.apply(lambda x: 1 if x['consec_gro'].shift() > 3 and x['dir_change'] == True else 0, axis =1)

df['regime'] = df.apply(lambda x: 2 if x['gdp_dir'] == 1 and x['cpi_dir'] == 1 else \
                                  (1 if x['gdp_dir'] == 1 and x['cpi_dir'] == -1 else \
                                   (3 if x['gdp_dir'] == -1 and x['cpi_dir'] == 1 else 4)), axis = 1)

df['vis'] = df.apply(lambda x: 2 if x['gdp_dir'] == 1 and x['cpi_dir'] == 1 else \
                                  (1 if x['gdp_dir'] == 1 and x['cpi_dir'] == -1 else \
                                   (-1 if x['gdp_dir'] == -1 and x['cpi_dir'] == 1 else -2)), axis = 1)







trace = go.Bar(
                        x = df['gdp_dir'][-80:].index,
                        y = df['gdp_dir'][-80:].values,
                        name = 'Regime',
                        #mode='markers',
                        marker=dict(
                                    #size=10,
                                    color = list(range(1,len(df['gdp_dir'].values))), #set color equal to a variable
                                    colorscale='Viridis',
                                    showscale=True),
                        opacity = 0.25
)
                        


trace1 = go.Scatter(
                        x = df['GDP CYOY Index'][-80:].index,
                        y = df['GDP CYOY Index'][-80:].rolling(window=2).mean(),
                        name = '2SMA',
                        mode='lines',
                        marker=dict(
                                    size=16,
                                    color = np.random.randn(500), #set color equal to a variable
                                    colorscale='Viridis',
                                    showscale=True
    )
)

trace2 = go.Scatter(
                        x = df['GDP CYOY Index'][-80:].index,
                        y = df['GDP CYOY Index'][-80:].ewm(3).mean(),
                        name = '3EMA',
                        mode='lines',
                        marker=dict(
                                    size=16,
                                    color = np.random.randn(500), #set color equal to a variable
                                    colorscale='Viridis',
                                    showscale=True
    )
)                        
                        
trace0 = go.Bar(
                        x = df['GDP CYOY Index'][-80:].index,
                        y = df['GDP CYOY Index'][-80:].values,
                        name = 'YoY Growth',
                        yaxis = 'y',
                        marker = dict(color = ('#a6a6a6')),
#                        line = dict(
#                                    color = ('#ccccff'),
#                                    width = 1.0,
#                                    ),
#                        fill = 'tonexty',
                        opacity = 1,
                       
                        
    
    )
  
                                    
#trace1 = go.Bar(        x = df['eq_direction'].index,
#                        y = df['eq_direction'].values,
#                        name = 'eq directions',
#                        yaxis = 'y',
#                        marker = dict(color = ('#1aff1a')),
##                        line = dict(
##                                    color = ('#1aff1a'),
##                                    width = 1.0,
##                                    ),
#                        #fill = 'tonexty',
#                        opacity = 0.50,
#        )
        
layout  = {'title' : f'4 Regime',
                   'xaxis' : {'title' : 'Date', #'type': 'date',
                              'fixedrange': True},
                   'yaxis' : {'title' : 'Regime', 'fixedrange': True},
                   
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
    
data = [trace, trace0,trace1, trace2]
figure = go.Figure(data=data, layout=layout)
py.iplot(figure, filename = f'Regime')