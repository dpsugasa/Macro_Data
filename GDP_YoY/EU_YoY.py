# -*- coding: utf-8 -*-
"""
US Real PCE SA 2009 Dollars YoY; YoY Transform Series
Plot on Plotly

@author: dsugasa
"""

import pandas as pd
import tia.bbg.datamgr as dm
#import numpy as np
from datetime import datetime
import plotly
import plotly.plotly as py #for plotting
import plotly.offline as offline
import plotly.graph_objs as go
plotly.tools.set_credentials_file(username='dpsugasa', api_key='yuwwkc1sb0')
import plotly.tools as tls
tls.embed('https://plot.ly/~dpsugasa/1/')


# create a DataManager for simpler api access
mgr = dm.BbgDataManager()
# set dates, securities, and fields
start_date = '01/01/2010'
end_date = "{:%m/%d/%Y}".format(datetime.now())
IDs = ['EUGNEMUY Index' ]
sids = mgr[IDs]
field1 = ['PX LAST']

###build the dataframe
df = sids.get_historical(field1, start_date, end_date)
df.columns = df.columns.droplevel(-1)

trace1 = go.Bar(
            x = df.index,
            y = df['EUGNEMUY Index'],
            name='GDP % YoY',
            marker = dict(
                        color = '#999999',
                            line = dict(
                                    color = '#404040',
                                    width = 1.5)
                            )
                            )


        
layout  = {'title' : 'EU GDP% YoY',
                   'xaxis' : {'title' : 'Date',
                              'fixedrange' : True,
#                              'hoverformat' : '.3f',
                              'showgrid' : True},
                              'autosize' : True,
                    'bargap' : 0,
#                    'margin' : {'l' : 200,
#                                'r' : 200,
#                                't' : 100,
#                                'b'  : 100},
                    'paper_bgcolor' : 'rgb(248, 248, 255)',
                    'plot_bgcolor' : 'rgb(248, 248, 255)',
                    'yaxis' : {'fixedrange' : True,
#                               'hoverformat' : '.3f',
                               'showgrid' : True}}
#                    'shapes': [{'type': 'line',
#                                'xref': 'x',
#                                'yref': 'paper',
#                                'x0': 0,
#                                'y0': 0,
#                                'x1': 0,
#                                'y1': 1,
#                                'line': {
#                                        'color': 'black',
#                                        'width': 3,}}
                                      
                                      
                   
    
data = [trace1]
figure = go.Figure(data=data, layout=layout)
py.iplot(figure, filename = 'GDP/EU_YOY/GDP')

# create a DataManager for simpler api access
mgr = dm.BbgDataManager()
# set dates, securities, and fields
start_date = '01/01/2011'
end_date = "{:%m/%d/%Y}".format(datetime.now())
IDs = ['SX5E Index' , 'ITRXEXE Index']
sids2 = mgr[IDs]
field2 = ['PX LAST']

###build the dataframe
df2 = sids2.get_historical(field2, start_date, end_date)
df2.columns = df2.columns.droplevel(-1)

trace1 = go.Scatter(
            x = df2.index,
            y = df2['SX5E Index'],
            name='SX5E')
#            marker = dict(
#                        color = '#999999',
#                            line = dict(
#                                    color = '#404040',
#                                    width = 1.5)
                            
                            

trace2 = go.Scatter(
            x = df2.index,
            y = df2['ITRXEXE Index'],
            name='iTraxx', 
            yaxis = 'y2')
#            marker = dict(
#                        color = '#999999',
#                            line = dict(
#                                    color = '#404040',
#                                    width = 1.5)
                            
                            


        
layout  = {'title' : 'EU Equity and Credit',
                   'xaxis' : {'title' : 'Date',
                              'fixedrange' : True,
#                              'hoverformat' : '.3f',
                              'showgrid' : True},
                              'autosize' : True,
                    'bargap' : 0,
#                    'margin' : {'l' : 200,
#                                'r' : 200,
#                                't' : 100,
#                                'b'  : 100},
                    'paper_bgcolor' : 'rgb(248, 248, 255)',
                    'plot_bgcolor' : 'rgb(248, 248, 255)',
                    'yaxis' : {'fixedrange' : True,
#                               'hoverformat' : '.3f',
                               'showgrid' : True},
                    'yaxis2' : {'fixedrange' : True,
#                               'hoverformat' : '.3f',
                               'showgrid' : True,
                               'overlaying' : 'y',
                               'side' : 'right'}
                    }
#                    'shapes': [{'type': 'line',
#                                'xref': 'x',
#                                'yref': 'paper',
#                                'x0': 0,
#                                'y0': 0,
#                                'x1': 0,
#                                'y1': 1,
#                                'line': {
#                                        'color': 'black',
#                                        'width': 3,}}
                                      
                                      
                   
    
data = [trace1, trace2]
figure = go.Figure(data=data, layout=layout)
py.iplot(figure, filename = 'GDP/EU_YOY/EU_Assets')




