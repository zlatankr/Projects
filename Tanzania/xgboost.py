# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 16:22:17 2017

@author: User
"""


# coding: utf-8

# In[1]:

import os
import pandas as pd
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import feature_process_helper


# ### XGBoost 1

# #### Load data

# In[2]:

X_train = pd.read_csv('X_train.csv')
X_test = pd.read_csv('X_test.csv')


# In[3]:

y_train = pd.read_csv('y_train.csv')
del y_train['id']


# In[4]:

y_train['status_group'] = y_train['status_group'].replace({'functional': 0, 'non functional': 1, 'functional needs repair': 2})


# #### Transform data


X_train, X_test = feature_process_helper.dates(X_train, X_test)
X_train, X_test = feature_process_helper.construction(X_train, X_test)
X_train, X_test = feature_process_helper.bools(X_train, X_test)
X_train, X_test = feature_process_helper.locs(X_train, X_test)
X_train, X_test = feature_process_helper.removal(X_train, X_test)
X_train, X_test = feature_process_helper.dummies(X_train, X_test)


xgdmat = xgb.DMatrix(X_train.values, y_train.values) # Create our DMatrix to make XGBoost more efficient

our_params = {'eta': 0.1, 'seed':0, 'subsample': 0.8, 'colsample_bytree': 0.8, 
             'objective': 'multi:softmax', 'max_depth':3, 'min_child_weight':1, 'num_class': 3} 

# Grid Search CV optimized settings

cv_xgb = xgb.cv(params = our_params, dtrain = xgdmat, num_boost_round = 300, nfold = 3,
                metrics = 'merror', # Make sure you enter metrics inside a list or you may encounter issues!
                early_stopping_rounds = 100) # Look for early stopping that minimizes error

cv_xgb