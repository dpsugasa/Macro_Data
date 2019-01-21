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

start_date = '01/01/1948'
end_date = "{:%m/%d/%Y}".format(datetime.now())
IDs = ['GDP CYOY Index']
fields = ['LAST PRICE']

df = LocalTerminal.get_historical(IDs, fields, start_date, end_date, period = 'QUARTERLY',
                                         non_trading_day_fill_option = 'ALL_CALENDAR_DAYS',
                                         non_trading_day_fill_method = 'PREVIOUS_VALUE').as_frame()
df.columns = df.columns.droplevel(-1)
#df = df.dropna()


#d_fx.columns = d_fx.columns.droplevel(-1)
