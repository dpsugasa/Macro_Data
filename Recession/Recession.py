# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 10:14:30 2018

Recession prediction model from a former colleague's LinkedIn post


@author: dpsugasa
"""

import pandas as pd
import datetime
from functools import reduce
import operator
import numpy as np
import pandas_datareader as pdr
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import matplotlib.pyplot as plt
 



def create_diff(column, periods):
    return column.diff(periods)
    
#create list of day ranges
days = range(5, 130, 5)
days = list(days)
 

# list of NBER recessions
recessions = [pd.date_range(datetime.date(1960, 4, 1), datetime.date(1961, 2, 28)),
              pd.date_range(datetime.date(1969, 12, 1),
                            datetime.date(1970, 11, 30)),
              pd.date_range(datetime.date(1973, 11, 1),
                            datetime.date(1975, 3, 31)),
              pd.date_range(datetime.date(1980, 1, 1),
                            datetime.date(1980, 7, 31)),
              pd.date_range(datetime.date(1981, 7, 1),
                            datetime.date(1982, 11, 30)),
              pd.date_range(datetime.date(1990, 7, 1),
                            datetime.date(1991, 3, 31)),
              pd.date_range(datetime.date(2001, 3, 1),
                            datetime.date(2001, 11, 30)),
              pd.date_range(datetime.date(2007, 12, 1), datetime.date(2009, 6, 30))]


# all dates in the time period
start_date = datetime.date(1962, 1, 2)
end_date = datetime.date.today()

date_list = pd.date_range(start_date, end_date)

# create a mask
mask = [date_list.isin(x) for x in recessions]
mask = reduce(operator.or_, mask)

# create our data frame
data = pd.Categorical(np.zeros(len(date_list)),
                      dtype="category", categories=[0, 1])
data = pd.Series(data, index=date_list)
data.name = 'Recession'
data[mask] = 1
data = pd.to_numeric(data)

#date_list = pd.date_range(start_date, end_date)

# now lets get some data from FRED
ten_yr = pdr.DataReader('DGS10', 'fred', start_date, end_date)
bills = pdr.DataReader('DTB3', 'fred', start_date, end_date)
ffunds = pdr.DataReader('DFF', 'fred', start_date, end_date)
rates = pd.concat([ffunds, bills, ten_yr], axis=1).fillna(method='ffill')
rates = rates / 100

# Add the curve
rates['Curve'] = rates['DGS10'] - rates['DTB3']




for z in rates.columns:
    for i in days:
        rates[f'{z}_{i}'] = create_diff(rates[z], i)
        
rates['recession'] = data
rates = rates.fillna(method = 'ffill').dropna()


X, y = rates.values[:, 4:103], rates.values[:, 104]

train_split = int(len(rates)*0.60)

X_train, X_test = X[0:train_split], X[train_split:]
y_train, y_test = y[0:train_split], y[train_split:]

# create dataset for lightgbm
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

# specify your configurations as a dict
params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'l2', 'l1'},
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}

print('Starting training...')

evals_result = {}
feature_name = list(rates.columns[4:103])

# train
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=100,
                valid_sets=lgb_eval,
                feature_name = feature_name,
                #categorical_feature=[21],
                evals_result=evals_result,
                verbose_eval=0)
#               early_stopping_rounds=50)



print('Saving model...')
# save model to file
gbm.save_model('model.txt')

print('Starting predicting...')
# predict
#y_pred_train = gbm.predict(X_train, num_iteration=gbm.best_iteration)
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
# eval
print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)

fin_df = pd.DataFrame(y_pred, columns = ['y_pred'], index = rates.index[train_split:])
fin_df['actual'] = rates['recession'][train_split:]

def render_plot_importance(importance_type, max_features=10,
                           ignore_zero=True, precision=4):
    ax = lgb.plot_importance(gbm, importance_type=importance_type,
                             max_num_features=max_features,
                             ignore_zero=ignore_zero, figsize=(12, 8),
                             precision=precision)
    plt.show()
    
render_plot_importance(importance_type='split')


##find best parameters
#def Grid_Search_CV_RFR(X_train, y_train):
#    reg = RandomForestClassifier()
#    param_grid = { 
#            "n_estimators"      : [10,25,50,100,500],
#            "max_features"      : ["auto"],
#            "min_samples_leaf" : [1,5,10,25,50,100]
#            }
#
#    tss_splits = TimeSeriesSplit(n_splits=10).split(X_train)
#    grid = GridSearchCV(reg, param_grid, cv=tss_splits, verbose=1) 
#
#    grid.fit(X_train, y_train)
#
#    return grid.best_score_ , grid.best_params_
#
#best_score, best_params = Grid_Search_CV_RFR(X_train, y_train)
#
#mf = best_params['max_features']
#msl = best_params['min_samples_leaf']
#ne = best_params['n_estimators']
#
#rfr = RandomForestRegressor(n_estimators=ne, max_features=mf, min_samples_leaf=msl, random_state=1)
#rfr.fit(X_train, y_train)
#
##date_list = pd.date_range(start_date, end_date)