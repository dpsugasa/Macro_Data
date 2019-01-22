# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 14:04:28 2019

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

start_date = '01/01/1990'
end_date = "{:%m/%d/%Y}".format(datetime.now())
IDs = ['GDP CYOY Index', 'CPI YOY Index', 'SPXT Index']
fields = ['LAST PRICE']

df = LocalTerminal.get_historical(IDs, fields, start_date, end_date, period = 'DAILY',
                                         non_trading_day_fill_option = 'ALL_CALENDAR_DAYS',
                                         non_trading_day_fill_method = 'PREVIOUS_VALUE').as_frame()
df.columns = df.columns.droplevel(-1)
df['SPXT Index'] = df['SPXT Index'].pct_change(252)
df =  df.resample('M').last()
#df = df.interpolate(method='linear')

df['gdp_ror'] = df['GDP CYOY Index'].pct_change()
df['cpi_ror'] = df['CPI YOY Index'].pct_change()

#df['SPXT Index'] = df['SPXT Index'].pct_change()

df['eq_direction']  = df.apply(lambda x: 1 if x['SPXT Index'] > 0 else(-1 if \
                              x['SPXT Index'] < 0 else 0), axis = 1)

df['gdp_direction'] = df.apply(lambda x: 1 if x['gdp_ror'] > 0 else(-1 if \
                              x['gdp_ror'] < 0 else 0), axis = 1)

df['cpi_direction'] = df.apply(lambda x: 1 if x['cpi_ror'] > 0 else(-1 if \
                              x['cpi_ror'] < 0 else 0), axis = 1)

#df['direction'] = df['gdp_direction'] + df['cpi_direction']


trace = go.Scatter(
                        x = df['cpi_direction'].values,
                        y = df['gdp_direction'].values,
                        name = 'GDP',
                        mode='markers',
                        marker=dict(
                                    size=10,
                                    color = list(range(1,len(df['gdp_direction'].values))), #set color equal to a variable
                                    colorscale='Viridis',
                                    showscale=True
    )
)

#trace1 = go.Scatter(
#                        #x = df['gdp_direction'].index,
#                        y = df['cpi_direction'].values,
#                        name = 'CPI',
#                        mode='markers',
#                        marker=dict(
#                                    size=16,
#                                    color = np.random.randn(500), #set color equal to a variable
#                                    colorscale='Viridis',
#                                    showscale=True
#    )
#)
#trace0 = go.Bar(
#                        x = df['cpi_direction'].index,
#                        y = df['cpi_direction'].values,
#                        name = 'inflation direction',
#                        yaxis = 'y',
#                        marker = dict(color = ('#a6a6a6')),
##                        line = dict(
##                                    color = ('#ccccff'),
##                                    width = 1.0,
##                                    ),
##                        fill = 'tonexty',
#                        opacity = 1,
#                       
#                        
#    
#    )
#  
#                                    
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
                   'xaxis' : {'title' : 'Direction', #'type': 'date',
                              'fixedrange': True},
                   'yaxis' : {'title' : 'Direction', 'fixedrange': True},
                   
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
    
data = [trace, trace1]
figure = go.Figure(data=data, layout=layout)
py.iplot(figure, filename = f'Growth Direction')


                              

#d[name].apply(lambda x: "Long" if x['LAST PRICE'] > x['MA_30'] else "Short", axis=1)
#df = df.dropna()


#d_fx.columns = d_fx.columns.droplevel(-1)
