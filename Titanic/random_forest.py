
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import feature_process_helper
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
get_ipython().magic(u'matplotlib inline')


# # RF Model 14

# #### Load data & transform variables


train = pd.read_csv(os.path.join('data', 'train.csv'))
test = pd.read_csv(os.path.join('data', 'test.csv'))
train, test = feature_process_helper.names(train, test)
train, test = feature_process_helper.age_impute(train, test)
train, test = feature_process_helper.cabin_num(train, test)
train, test = feature_process_helper.cabin(train, test)
train, test = feature_process_helper.embarked_impute(train, test)
train, test = feature_process_helper.fam_size(train, test)
train['Ticket_Len'] = train['Ticket'].apply(lambda x: len(x))
test['Ticket_Len'] = test['Ticket'].apply(lambda x: len(x))
train, test = feature_process_helper.ticket_grouped(train, test)
train, test = feature_process_helper.dummies(train, test, columns = ['Pclass', 'Sex', 'Embarked', 'Ticket_Lett',
                                                                     'Cabin_Letter', 'Name_Title', 'Fam_Size'])
train, test = feature_process_helper.drop(train, test, bye = ['Ticket', 'SibSp', 'Parch'])


# #### Tune hyper-parameters


rf14 = RandomForestClassifier(max_features='auto',
                                oob_score=True,
                                random_state=1,
                                n_jobs=-1)

param_grid = { "criterion"   : ["gini", "entropy"],
             "min_samples_leaf" : [1,5,10],
             "min_samples_split" : [2, 4, 10, 12, 16],
             "n_estimators": [50, 100, 400, 700, 1000]}

gs = GridSearchCV(estimator=rf14,
                  param_grid=param_grid,
                  scoring='accuracy',
                  cv=3,
                  n_jobs=-1)

gs = gs.fit(train.iloc[:, 2:], train.iloc[:, 1])



print(gs.best_score_)
print(gs.best_params_)
#print(gs.cv_results_)


# #### Fit model

rf14 = RandomForestClassifier(criterion='gini', 
                             n_estimators=700,
                             min_samples_split=10,
                             min_samples_leaf=1,
                             max_features='auto',
                             oob_score=True,
                             random_state=1,
                             n_jobs=-1)
rf14.fit(train.iloc[:, 2:], train.iloc[:, 1])
print "%.4f" % rf14.oob_score_ 


# #### Obtain cross-validation score with optimal hyperparameters

scores1 = cross_val_score(rf14, train.iloc[:, 2:], train.iloc[:, 1], n_jobs=-1)
scores1.mean()


# #### Inspect feature ranking


pd.concat((pd.DataFrame(train.iloc[:, 2:].columns, columns = ['variable']), 
           pd.DataFrame(rf14.feature_importances_, columns = ['importance'])), 
          axis = 1).sort_values(by='importance', ascending = False)[:20]


# #### Generate submission file


test['Fare'].fillna(train['Fare'].mean(), inplace = True)
predictions = rf14.predict(test.iloc[:, 1:])
predictions = pd.DataFrame(predictions, columns=['Survived'])
predictions = pd.concat((test.iloc[:, 0], predictions), axis = 1)
predictions.to_csv(os.path.join('submission_files', 'y_test14.csv'), sep=",", index = False)


# Leaderboard score: 0.82775   
# 150 out of 6048
