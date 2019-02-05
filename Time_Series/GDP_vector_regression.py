# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 11:14:24 2019

@author: dpsugasa
"""

import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# Forecasting years, number of years to use for out-of-sample 
# evaluation (We will create one forecast per country for each year
# which we compare to the actual GDP)
nyears = 10
# Number of lags to use for GP regression
lags = 5
# Indicator labels and names in the World Bank API
indicators  = {"gdp"        : "NY.GDP.MKTP.CD",
               "population" :    "SP.POP.TOTL",
               "inflation"  : "FP.CPI.TOTL.ZG"}
nindicators = len(indicators)
# The variable to forecast, should be one of the indicator labels
target_variable = "gdp"
# Countries to include in the data, specified as ISO country codes
countries  = ['au','ca','de','es','fr','gb','jp','us']
ncountries = len(countries)
# Start and end year for the data set
start_year = 1976
end_year   = 2016


template_url = "http://api.worldbank.org/v2/countries/{0}/indi"
template_url +="cators/{1}?date={2}:{3}&format=json&per_page=999"
# Countries should be ISO identifiers separated by semi-colon
country_str = ';'.join(countries)
raw_data = pd.DataFrame()
for label, indicator in indicators.items():
# Fill in the template URL
    url = template_url.format(country_str, indicator, 
                                  start_year, end_year)
    
    # Request the data
    json_data = requests.get(url)
    
    # Convert the JSON string to a Python object
    json_data = json_data.json()
    
    # The result is a list where the first element is meta-data, 
    # and the second element is the actual data
    json_data = json_data[1]
    
    # Loop over all data points, pick out the values and append 
    # them to the data frame
    for data_point in json_data:
        
        country = data_point['country']['id']
        
        # Create a variable for each country and indicator pair
        item    = country + '_' + label
        
        year    = data_point['date']
        
        value   = data_point['value']
        
        # Append to data frame
        new_row  = pd.DataFrame([[item, year, value]],
                                columns=['item', 'year', 'value'])
        raw_data = raw_data.append(new_row)
# Pivot the data to get unique years along the columns,
# and variables along the rows
raw_data = raw_data.pivot('year', 'item', 'value')
# Let's look at the first few rows and columns
print('\n', raw_data.iloc[:10, :5], '\n')

for lab in indicators.keys():
    
    indicator = raw_data[[x for x in raw_data.columns 
                              if x.split("_")[-1] == lab]]
    indicator.plot(title=lab)
    plt.show()
    
    
# Calculate rates of change instead of absolute levels
# (Runtime warning expected due to NaN)
data = np.log(raw_data).diff().iloc[1:,:]
# Set NaN to zero
data.fillna(0, inplace=True)
# Subtract the mean from each series
data = data - data.mean()
# Convert to date type
data.index = pd.to_datetime(data.index, format='%Y')
# Put the target variable into a separate data frame
target = data[[x for x in data.columns 
                   if x.split("_")[-1] == target_variable]]

errors = target.iloc[-nyears:] - target.shift().iloc[-nyears:]
# Root mean squared error
rmse = errors.pow(2).sum().sum()/(nyears*ncountries)**.5
print('\n\t' + '-' * 18)
print("\t| Error: ", np.round(rmse, 4), '|')
print('\t' + '-' * 18 + '\n')

from statsmodels.tsa.api import VAR
# Sum of squared errors
sse = 0
for t in range(nyears):
    
    # Create a VAR model
    model = VAR(target.iloc[t:-nyears+t], freq='AS')
    #model = model.select_order(24)
    
    # Estimate the model parameters
    results = model.fit(maxlags=1)
    
    actual_values = target.values[-nyears+t+1]
    
    forecasts = results.forecast(target.values[:-nyears+t], 1)
    forecasts = forecasts[0,:ncountries]
sse += ((actual_values - forecasts)**2).sum()
# Root mean squared error
rmse = (sse / (nyears * ncountries))**.5
print('\n\t' + '-' * 18)
print("\t| Error: ", np.round(rmse, 4), '|')
print('\t' + '-' * 18 + '\n')

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
# We set the parameter of the covariance function to
# approximately equal to the median distance between data points,
# a common heuristic for this covariance function. The 'alpha' 
# argument is the noise variance, which we set equal to the 
# covariance parameter.
gpr = GaussianProcessRegressor(kernel=RBF(0.1), alpha=0.1)
# Number of data points for estimation/fitting for each forecast
ndata = target.shape[0] - nyears - lags
# Sum of squared errors
sse = 0
for t in range(nyears):
    
    # Observations for the target variables
    y = np.zeros((ndata, ncountries))
# Observations for the independent variables
    X = np.zeros((ndata, lags*ncountries*nindicators))
    
    for i in range(ndata):
        
        y[i] = target.iloc[t+i+1]
        X[i] = data.iloc[t+i+2:t+i+2+lags].values.flatten()
        
    gpr.fit(X, y)
    
    x_test   = np.expand_dims(data.iloc[t+1:t+1+lags].values.flatten(), 0)
    forecast = gpr.predict(x_test)
    
    sse += ((target.iloc[t].values - forecast)**2).sum()
    
rmse = (sse / (nyears * ncountries))**.5
print('\n\t' + '-' * 18)
print("\t| Error: ", np.round(rmse, 4), '|')
print('\t' + '-' * 18 + '\n')