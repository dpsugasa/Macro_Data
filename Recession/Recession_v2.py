# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 14:45:19 2018

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
import json
try:
    import cPickle as pickle
except BaseException:
    import pickle



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
num_feature = 99

train_split = int(len(rates)*0.60)

X_train, X_test = X[0:train_split], X[train_split:]
y_train, y_test = y[0:train_split], y[train_split:]

# create dataset for lightgbm
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'binary_logloss',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}

# generate feature names
feature_name = ['feature_' + str(col) for col in range(num_feature)]

print('Starting training...')
# feature_name and categorical_feature
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=10,
                valid_sets=lgb_train,  # eval training data
                feature_name=feature_name,
                categorical_feature=[21])

print('Finished first 10 rounds...')
# check feature name
print('7th feature name is:', lgb_train.feature_name[6])

print('Saving model...')
# save model to file
gbm.save_model('model.txt')

print('Dumping model to JSON...')
# dump model to JSON (and save to file)
model_json = gbm.dump_model()

with open('model.json', 'w+') as f:
    json.dump(model_json, f, indent=4)

# feature names
print('Feature names:', gbm.feature_name())

# feature importances
print('Feature importances:', list(gbm.feature_importance()))

print('Loading model to predict...')
# load model to predict
bst = lgb.Booster(model_file='model.txt')
# can only predict with the best iteration (or the saving iteration)
y_pred = bst.predict(X_test)
# eval with loaded model
print("The rmse of loaded model's prediction is:", mean_squared_error(y_test, y_pred) ** 0.5)

print('Dumping and loading model with pickle...')
# dump model with pickle
with open('model.pkl', 'wb') as fout:
    pickle.dump(gbm, fout)
# load model with pickle to predict
with open('model.pkl', 'rb') as fin:
    pkl_bst = pickle.load(fin)
# can predict with any iteration when loaded in pickle way
y_pred = pkl_bst.predict(X_test, num_iteration=7)
# eval with loaded model
print("The rmse of pickled model's prediction is:", mean_squared_error(y_test, y_pred) ** 0.5)

# continue training
# init_model accepts:
# 1. model file name
# 2. Booster()
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=10,
                init_model='model.txt',
                valid_sets=lgb_eval)

print('Finished 10 - 20 rounds with model file...')

# decay learning rates
# learning_rates accepts:
# 1. list/tuple with length = num_boost_round
# 2. function(curr_iter)
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=10,
                init_model=gbm,
                learning_rates=lambda iter: 0.05 * (0.99 ** iter),
                valid_sets=lgb_eval)

print('Finished 20 - 30 rounds with decay learning rates...')

# change other parameters during training
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=10,
                init_model=gbm,
                valid_sets=lgb_eval,
                callbacks=[lgb.reset_parameter(bagging_fraction=[0.7] * 5 + [0.6] * 5)])

print('Finished 30 - 40 rounds with changing bagging_fraction...')


# self-defined objective function
# f(preds: array, train_data: Dataset) -> grad: array, hess: array
# log likelihood loss
def loglikelihood(preds, train_data):
    labels = train_data.get_label()
    preds = 1. / (1. + np.exp(-preds))
    grad = preds - labels
    hess = preds * (1. - preds)
    return grad, hess


# self-defined eval metric
# f(preds: array, train_data: Dataset) -> name: string, eval_result: float, is_higher_better: bool
# binary error
def binary_error(preds, train_data):
    labels = train_data.get_label()
    return 'error', np.mean(labels != (preds > 0.5)), False


gbm = lgb.train(params,
                lgb_train,
                num_boost_round=10,
                init_model=gbm,
                fobj=loglikelihood,
                feval=binary_error,
                valid_sets=lgb_eval)

print('Finished 40 - 50 rounds with self-defined objective function and eval metric...')

print('Starting a new training job...')


# callback
def reset_metrics():
    def callback(env):
        lgb_eval_new = lgb.Dataset(X_test, y_test, reference=lgb_train)
        if env.iteration - env.begin_iteration == 5:
            print('Add a new valid dataset at iteration 5...')
            env.model.add_valid(lgb_eval_new, 'new_valid')
    callback.before_iteration = True
    callback.order = 0
    return callback


gbm = lgb.train(params,
                lgb_train,
                num_boost_round=10,
                valid_sets=lgb_train,
                callbacks=[reset_metrics()])

print('Finished first 10 rounds with callback function...')