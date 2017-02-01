# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 16:22:17 2017

@author: User
"""

import os
import pandas as pd
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
os.chdir('C:\\Users\\User\\Dropbox\\Documents\\Analytics\\Projects\\Tanzania')
import feature_process_helper
import numpy as np

# Load data

X_train = pd.read_csv('X_train.csv')
X_test = pd.read_csv('X_test.csv')

y_train = pd.read_csv('y_train.csv')
del y_train['id']

y_train['status_group'] = y_train['status_group'].replace({'functional': 0, 'non functional': 1, 'functional needs repair': 2})


# Transform data

X_train, X_test = feature_process_helper.dates(X_train, X_test)
X_train, X_test = feature_process_helper.dates2(X_train, X_test)
X_train, X_test = feature_process_helper.construction(X_train, X_test)
X_train, X_test = feature_process_helper.bools(X_train, X_test)
X_train, X_test = feature_process_helper.locs(X_train, X_test)
X_train['population'] = np.log(X_train['population'])
X_test['population'] = np.log(X_test['population'])
X_train, X_test = feature_process_helper.removal(X_train, X_test)
X_train, X_test = feature_process_helper.lda(X_train, X_test, y_train, 
                                             cols = ['gps_height', 'latitude', 'longitude'])
X_train, X_test = feature_process_helper.dummies(X_train, X_test)

# Tune parameters

cv_params = {'max_depth': [3,5,7], 'min_child_weight': [1,3,5]}

ind_params = {'learning_rate': 0.1, 'n_estimators': 100, 'seed':0,
              'subsample': 0.8, 'colsample_bytree': 0.8, 
             'objective': 'multi:softmax'}

# Optimize for accuracy since that is the metric used in the Adult Data Set notation
optimized_GBM = GridSearchCV(xgb.XGBClassifier(**ind_params), cv_params, 
                             scoring = 'accuracy', cv = 3, n_jobs = -1) 

# can i skip this step???
optimized_GBM.fit(X_train, y_train.values.ravel())

"""

gsearch1 = GridSearchCV(cv=5, error_score='raise',
             estimator=XGBClassifier(base_score=0.5, colsample_bylevel=1, colsample_bytree=0.8,
                                     gamma=0, learning_rate=0.1, max_delta_step=0, max_depth=3,
                                     min_child_weight=1, missing=None, n_estimators=100, nthread=-1,
                                     objective='multi:softmax', reg_alpha=0, reg_lambda=1,
                                     scale_pos_weight=1, seed=0, silent=True, subsample=0.8),
                                     fit_params={}, iid=True, n_jobs=-1,
                                     param_grid={'min_child_weight': [1, 3, 5], 'max_depth': [3, 5, 7]},
                                     pre_dispatch='2*n_jobs', refit=True, scoring='accuracy', verbose=0)

"""

# Build final model

xgdmat = xgb.DMatrix(X_train.values, y_train.values) # Create our DMatrix to make XGBoost more efficient

our_params = {'eta': 0.5, 'seed':0, 'subsample': 0.8, 'colsample_bytree': 0.8, 
             'objective': 'multi:softmax', 'max_depth':3, 'min_child_weight':1, 'num_class': 3} 

# Grid Search CV optimized settings

cv_xgb = xgb.cv(params = our_params, dtrain = xgdmat, num_boost_round = 50, nfold = 2,
                metrics = 'merror', # Make sure you enter metrics inside a list or you may encounter issues!
                early_stopping_rounds = 100) # Look for early stopping that minimizes error

cv_xgb