# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 21:45:18 2019

@author: dpsugasa
"""

import numpy as np
import pandas
import statsmodels.api as sm
from statsmodels.tsa.api import VAR, DynamicVAR


mdata = sm.datasets.macrodata.load_pandas().data
dates = mdata[['year', 'quarter']].astype(int).astype(str)
quarterly = dates["year"] + "Q" + dates["quarter"]
from statsmodels.tsa.base.datetools import dates_from_str
quarterly = dates_from_str(quarterly)
mdata = mdata[['realgdp','realcons','realinv']]
mdata.index = pandas.DatetimeIndex(quarterly)
data = np.log(mdata).diff().dropna()
model = VAR(data)

results = model.fit(2)
results.summary()

results.plot()

results.plot_acorr()

model.select_order(15)

results = model.fit(maxlags=15, ic='aic')

lag_order = results.k_ar
results.forecast(data.values[-lag_order:], 5)

results.plot_forecast(5)