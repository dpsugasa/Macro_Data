# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 13:16:31 2019

Highlighting periods of high SPX volatility

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
from random import randint

fred = credentials.fred

#set the script start time
start_time = datetime.now()
date_now =  "{:%m_%d_%Y}".format(datetime.now())

start_date = '01/01/1950'
end_date = "{:%m/%d/%Y}".format(datetime.now())
IDs = ['SPX Index'] 
fields = ['LAST PRICE', 'VAR_SWAP_12M_LV','12MTH_IMPVOL_100.0%MNY_DF']

df = LocalTerminal.get_historical(IDs, fields, start_date, end_date).as_frame() #period = 'QUARTERLY',
                                         #non_trading_day_fill_option = 'ALL_CALENDAR_DAYS',
                                         #non_trading_day_fill_method = 'PREVIOUS_VALUE').as_frame()
df.columns = df.columns.droplevel()
#df = df.resample('Q').mean()
df = df.dropna()

spikes = df['VAR_SWAP_12M_LV'][df['VAR_SWAP_12M_LV'] > (df['VAR_SWAP_12M_LV'].rolling(window = 90).mean() + \
            df['VAR_SWAP_12M_LV'].rolling(window=90).std())]

trace = go.Bar(
                        x = spikes.index,
                        y = spikes.values,
                        name = '12M Var Swap',
                        #mode='markers',
                        marker=dict(
                                    
                                    color = [randint(1, len(spikes)) for x in range(1,len(spikes.values))], #set color equal to a variable
                                    colorscale='Jet',
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
        
layout  = {'title' : f'SPX 12M VarSwap 1sd Above the 90d Mean',
                   'xaxis' : {'title' : 'Date', #'type': 'date',
                              'fixedrange': True},
                   'yaxis' : {'title' : 'Implied Vol', 'fixedrange': True},
                   'barmode' : 'group'
                   
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
    
data = [trace]
figure = go.Figure(data=data, layout=layout)
py.iplot(figure, filename = f'Flares/SPX_Variance_Levels')

print ("Time to complete:", datetime.now() - start_time)

