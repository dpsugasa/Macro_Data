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

df['consec_gro'] = df['gdp_dir'] * (df['gdp_dir'].groupby((df['gdp_dir'] != df['gdp_dir'].shift()).cumsum()).cumcount()+1)
df['dir_change'] = df['gdp_dir'] != df['gdp_dir'].shift()
df['vol_watch'] = np.where(df['consec_gro'].shift() > 2 , 1,0)
df['danger'] = df.apply(lambda x: 1 if x['vol_watch'] ==1 and x['dir_change'] == True else 0, axis =1)
#df['vol_watch'] = df.apply(lambda x: 1 if x['consec_gro'].shift() > 3 and x['dir_change'] == True else 0, axis =1)

df['regime'] = df.apply(lambda x: 2 if x['gdp_dir'] == 1 and x['cpi_dir'] == 1 else \
                                  (1 if x['gdp_dir'] == 1 and x['cpi_dir'] == -1 else \
                                   (3 if x['gdp_dir'] == -1 and x['cpi_dir'] == 1 else 4)), axis = 1)

df['vis'] = df.apply(lambda x: 2 if x['gdp_dir'] == 1 and x['cpi_dir'] == 1 else \
                                  (1 if x['gdp_dir'] == 1 and x['cpi_dir'] == -1 else \
                                   (-1 if x['gdp_dir'] == -1 and x['cpi_dir'] == 1 else -2)), axis = 1)




'''
#df['eq_direction']  = df.apply(lambda x: 1 if x['SPXT Index'] > 0 else(-1 if \
#                              x['SPXT Index'] < 0 else 0), axis = 1)
#
#df['gdp_direction'] = df.apply(lambda x: 1 if x['gdp_ror'] > 0 else(-1 if \
#                              x['gdp_ror'] < 0 else 0), axis = 1)
#
#df['cpi_direction'] = df.apply(lambda x: 1 if x['cpi_ror'] > 0 else(-1 if \
#                              x['cpi_ror'] < 0 else 0), axis = 1)
#
##df['direction'] = df['gdp_direction'] + df['cpi_direction']


trace = go.Bar(
                        x = df['gdp_dir'][-80:].index,
                        y = df['gdp_dir'][-80:].values,
                        name = 'Regime',
#                        mode='markers',
#                        marker=dict(
#                                    size=10,
#                                    color = list(range(1,len(df['gdp_dir'].values))), #set color equal to a variable
#                                    colorscale='Viridis',
#                                    showscale=True
#    )
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
    
data = [trace]
figure = go.Figure(data=data, layout=layout)
py.iplot(figure, filename = f'Regime')


start_date = '01/01/1995'
end_date = "{:%m/%d/%Y}".format(datetime.now())
IDs = ['HYG US Equity', 'NDX Index', 'SPX Index'] #'SPXT Index']
fields = ['LAST PRICE']

df_eq = LocalTerminal.get_historical(IDs, fields, start_date, end_date).as_frame()
df_eq.columns = df_eq.columns.droplevel(-1)   
df_eq = df_eq.fillna(method = 'ffill') #.resample('M').last()
df_eq = df_eq.pct_change() 
df_eq = df_eq.resample('Q').sum()           

frames = [df_eq, df]
baf = pd.concat(frames, join='outer', axis =1)
#baf = baf.fillna(method = 'ffill')   #removes last quarter; also discover if changes in prolonged periods or good markets precipitate higher vol

baf = baf.dropna()   

q4 = baf['HYG US Equity'][(baf['regime'] == 4)].dropna() #i think this is using the returns of zero
print(q4.mean())
print(q4.std())

q1 = baf['HYG US Equity'][(baf['regime'] == 1)].dropna()
print(q1.mean())
print(q1.std())

q3 = baf['HYG US Equity'][(baf['regime'] == 3)].dropna()
print(q3.mean())
print(q3.std())

q2 = baf['HYG US Equity'][(baf['regime'] == 2)].dropna()
print(q2.mean())
print(q2.std())

q_4 = baf['NDX Index'][(baf['regime'] == 4)].dropna()
print(q_4.mean())
print(q_4.std())

q_1 = baf['NDX Index'][(baf['regime'] == 1)].dropna()
print(q_1.mean())
print(q_1.std())

q_3 = baf['NDX Index'][(baf['regime'] == 3)].dropna()
print(q_3.mean())
print(q_3.std())

q_2 = baf['NDX Index'][(baf['regime'] == 2)].dropna()
print(q_2.mean())
print(q_2.std())

q_4_spx = baf['SPX Index'][(baf['regime'] == 4)].dropna()
print(q_4_spx.mean())
print(q_4_spx.std())

q_1_spx = baf['SPX Index'][(baf['regime'] == 1)].dropna()
print(q_1_spx.mean())
print(q_1_spx.std())

q_3_spx = baf['SPX Index'][(baf['regime'] == 3)].dropna()
print(q_3_spx.mean())
print(q_3_spx.std())

q_2_spx = baf['SPX Index'][(baf['regime'] == 2)].dropna()
print(q_2_spx.mean())
print(q_2_spx.std())

      

#d[name].apply(lambda x: "Long" if x['LAST PRICE'] > x['MA_30'] else "Short", axis=1)
#df = df.dropna()


#d_fx.columns = d_fx.columns.droplevel(-1)
'''