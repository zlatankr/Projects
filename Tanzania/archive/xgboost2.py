# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 14:02:05 2017

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

param_test1 = cv_params = {'max_depth': [3,5], 'min_child_weight': [1]}

gsearch1 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.5, n_estimators=50,
                                                  gamma=0, subsample=0.8, colsample_bytree=0.8,
                                                  objective= 'multi:softmax', nthread=4, scale_pos_weight=1, seed=27),
                                                  param_grid = param_test1, scoring='accuracy', iid=False, cv=2)

gsearch1.fit(X_train,y_train.values.ravel())

gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_


# Build final model

xgdmat = xgb.DMatrix(X_train.values, y_train.values) # Create our DMatrix to make XGBoost more efficient

our_params = {'eta': 0.1, 'seed':0, 'subsample': 0.8, 'colsample_bytree': 0.8, 
             'objective': 'multi:softmax', 'max_depth':3, 'min_child_weight':1, 'num_class': 3} 

# Grid Search CV optimized settings

cv_xgb = xgb.cv(params = our_params, dtrain = xgdmat, num_boost_round = 300, nfold = 3,
                metrics = 'merror', # Make sure you enter metrics inside a list or you may encounter issues!
                early_stopping_rounds = 100) # Look for early stopping that minimizes error

cv_xgb

########################
########################

xgb_model = xgb.XGBClassifier()

#when in doubt, use xgboost
parameters = {'nthread':[4], #when use hyperthread, xgboost may become slower
              'objective':['multi:softmax'],
              'learning_rate': [0.15], #so called `eta` value
              'max_depth': [8],
              'min_child_weight': [1, 3],
              'silent': [1],
              'subsample': [0.9],
              'colsample_bytree': [0.5],
              'n_estimators': [50], #number of trees
              'seed': [1337]}

#evaluate with roc_auc_truncated
"""
def _score_func(estimator, X, y):
    pred_probs = estimator.predict_proba(X)[:, 1]
    return roc_auc_truncated(y, pred_probs)
"""

#should evaluate by train_eval instead of the full dataset
clf = GridSearchCV(xgb_model, parameters, n_jobs=4, 
                   #cv=StratifiedKFold(train_eval['signal'], n_folds=5, shuffle=True), 
                   cv=2,
                    scoring='roc_auc',
                   verbose=2, refit=True)

clf.fit(X_train,y_train.values.ravel())

######
######

# STEP 1: fix learning rate and number of estimators

xgb1 = XGBClassifier(
 learning_rate =0.25,
 n_estimators=1000,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27)

xgdmat = xgb.DMatrix(X_train.values, y_train.values)

modelfit(xgb1, X_train,y_train.values.ravel())

our_params = {'eta': 0.25, 'seed':0, 'subsample': 0.8, 'colsample_bytree': 0.8, 
             'objective': 'multi:softmax', 'max_depth':5, 'min_child_weight':1, 'num_class': 3} 

cv_xgb = xgb.cv(params = our_params, dtrain = xgdmat, num_boost_round = 1000, nfold = 3,
                metrics = 'merror', # Make sure you enter metrics inside a list or you may encounter issues!
                early_stopping_rounds = 50) # Look for early stopping that minimizes error

cv_xgb

### looks like our best iteration was at iteration 510...

# STEP 2: Tune max_depth and min_child_weight

param_test1 = cv_params = {'max_depth': [3, 5, 7], 'min_child_weight': [1, 3, 5]}

gsearch1 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.25, n_estimators=510,
                                                  gamma=0, subsample=0.8, colsample_bytree=0.8,
                                                  objective= 'multi:softmax', nthread=4, scale_pos_weight=1, seed=27),
                                                  param_grid = param_test1, scoring='accuracy', iid=False, cv=2)

gsearch1.fit(X_train,y_train.values.ravel())

gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_

# STEP 3: Tune gamma

## apparently, we want to tune the n_estimators after each hyperparameter tuning round...

param_test2 = cv_params = {'gamma':[i/10.0 for i in range(0,5)]}

gsearch2 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.25, n_estimators=510,
                                                  gamma=0, subsample=0.8, colsample_bytree=0.8,
                                                  objective= 'multi:softmax', nthread=4, scale_pos_weight=1, seed=27),
                                                  param_grid = param_test2, scoring='accuracy', iid=False, cv=2)

gsearch1.fit(X_train,y_train.values.ravel())

gsearch2.grid_scores_, gsearch2.best_params_, gsearch2.best_score_

# STEP 4: Tune subsample and colsample_bytree

param_test3 = cv_params = {'subsample':[i/10.0 for i in range(6,10)],
                                        'colsample_bytree':[i/10.0 for i in range(6,10)]}

gsearch3 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.25, n_estimators=510,
                                                  gamma=0, subsample=0.8, colsample_bytree=0.8,
                                                  objective= 'multi:softmax', nthread=4, scale_pos_weight=1, seed=27),
                                                  param_grid = param_test3, scoring='accuracy', iid=False, cv=2)

gsearch3.fit(X_train,y_train.values.ravel())

gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_

# STEP 5: Tuning Regularization Parameters

param_test4 = cv_params = {'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]}

gsearch4 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.25, n_estimators=510,
                                                  gamma=0, subsample=0.8, colsample_bytree=0.8,
                                                  objective= 'multi:softmax', nthread=4, scale_pos_weight=1, seed=27),
                                                  param_grid = param_test4, scoring='accuracy', iid=False, cv=2)

gsearch4.fit(X_train,y_train.values.ravel())

gsearch4.grid_scores_, gsearch4.best_params_, gsearch4.best_score_


## STEP 6: Build final model 
    
our_params = {'eta': 0.1, 'seed':0, 'subsample': 0.8, 'colsample_bytree': 0.8, 
             'objective': 'multi:softmax', 'max_depth':7, 'min_child_weight':1, 'num_class': 3} 

final_gb = xgb.train(our_params, xgdmat, num_boost_round = 510)

importances = final_gb.get_fscore()
importances

# STEP 7: Run final model on test data

testdmat = xgb.DMatrix(X_test.values)

# Predict using our testdmat
y_pred = final_gb.predict(testdmat)


y_test = pd.read_csv('y_test.csv')
pred = pd.DataFrame(y_pred, columns = [y_test.columns[1]])
pred['status_group'].replace({0: 'functional', 1: 'non functional', 2: 'functional needs repair'}, inplace=True)
del y_test['status_group']
y_test = pd.concat((y_test, pred), axis = 1)
y_test.to_csv(os.path.join('submission_files', 'y_test34.csv'), sep=",", index = False)
