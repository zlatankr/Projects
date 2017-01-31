

```python
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
%matplotlib inline
```

# RF Model 1

#### Load data


```python
train = pd.read_csv(os.path.join('data', 'train.csv'))
test = pd.read_csv(os.path.join('data', 'test.csv'))
```

#### Transform variables


```python
train, test = feature_process_helper.names(train, test)
train, test = feature_process_helper.age_impute(train, test)
train, test = feature_process_helper.cabin(train, test)
train, test = feature_process_helper.embarked_impute(train, test)
train, test = feature_process_helper.dummies(train, test)
train, test = feature_process_helper.drop(train, test)
```

#### Tune hyper-parameters


```python
rf1 = RandomForestClassifier(max_features='auto',
                                oob_score=True,
                                random_state=1,
                                n_jobs=-1)
```


```python
param_grid = { "criterion"   : ["gini", "entropy"],
             "min_samples_leaf" : [1,5,10],
             "min_samples_split" : [2, 4, 10, 12, 16],
             "n_estimators": [50, 100, 400, 700]}

gs = GridSearchCV(estimator=rf1,
                  param_grid=param_grid,
                  scoring='accuracy',
                  cv=3,
                  n_jobs=-1)

gs = gs.fit(train.iloc[:, 2:], train.iloc[:, 1])
```


```python
print(gs.best_score_)
print(gs.best_params_)
#print(gs.cv_results_)
```

    0.826038159371
    {'min_samples_split': 10, 'n_estimators': 100, 'criterion': 'gini', 'min_samples_leaf': 1}
    

#### Fit model


```python
rf1 = RandomForestClassifier(criterion='gini', 
                             n_estimators=100,
                             min_samples_split=10,
                             min_samples_leaf=1,
                             max_features='auto',
                             oob_score=True,
                             random_state=1,
                             n_jobs=-1)
rf1.fit(train.iloc[:, 2:], train.iloc[:, 1])
print "%.4f" % rf1.oob_score_ 
```

    0.8238
    

#### Obtain cross-validation score with optimal hyperparameters


```python
scores1 = cross_val_score(rf1, train.iloc[:, 2:], train.iloc[:, 1], n_jobs=-1)
scores1.mean()
```




    0.82603815937149283



#### Inspect feature ranking


```python
pd.concat((pd.DataFrame(train.iloc[:, 2:].columns, columns = ['variable']), 
           pd.DataFrame(rf1.feature_importances_, columns = ['importance'])), 
          axis = 1).sort_values(by='importance', ascending = False)[:20]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>variable</th>
      <th>importance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Fare</td>
      <td>0.142931</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Name_Title_Mr.</td>
      <td>0.131256</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Sex_male</td>
      <td>0.111903</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Name_Len</td>
      <td>0.107163</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Age</td>
      <td>0.098652</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Sex_female</td>
      <td>0.090083</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Pclass_3</td>
      <td>0.063281</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Name_Title_Mrs.</td>
      <td>0.037778</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Name_Title_Miss.</td>
      <td>0.035874</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Cabin_Letter_n</td>
      <td>0.034054</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Pclass_1</td>
      <td>0.023819</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Pclass_2</td>
      <td>0.022122</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Embarked_S</td>
      <td>0.017631</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Name_Title_Master.</td>
      <td>0.013617</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Embarked_C</td>
      <td>0.011758</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Age_Null_Flag</td>
      <td>0.011680</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Embarked_Q</td>
      <td>0.009113</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Cabin_Letter_E</td>
      <td>0.008164</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Cabin_Letter_B</td>
      <td>0.006776</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Cabin_Letter_D</td>
      <td>0.006237</td>
    </tr>
  </tbody>
</table>
</div>



#### Generate submission file


```python
test['Fare'].fillna(train['Fare'].mean(), inplace = True)
predictions = rf1.predict(test.iloc[:, 1:])
predictions = pd.DataFrame(predictions, columns=['Survived'])
predictions = pd.concat((test.iloc[:, 0], predictions), axis = 1)
predictions.to_csv(os.path.join('submission_files', 'y_test1.csv'), sep=",", index = False)
```

# RF Model 2

#### Load data


```python
train = pd.read_csv(os.path.join('data', 'train.csv'))
test = pd.read_csv(os.path.join('data', 'test.csv'))
```

#### Transform variables


```python
train, test = feature_process_helper.names(train, test)
train, test = feature_process_helper.age_impute(train, test)
train, test = feature_process_helper.cabin(train, test)
train, test = feature_process_helper.embarked_impute(train, test)
train, test = feature_process_helper.dummies(train, test, columns = ['Pclass', 'Sex', 'Embarked', 'Cabin_Letter', 'Name_Title',
                                                                    'SibSp', 'Parch'])
train, test = feature_process_helper.drop(train, test, bye = ['Ticket'])
```

#### Tune hyperparameters


```python
rf1 = RandomForestClassifier(max_features='auto',
                                oob_score=True,
                                random_state=1,
                                n_jobs=-1)
```


```python
param_grid = { "criterion"   : ["gini", "entropy"],
             "min_samples_leaf" : [1,5,10],
             "min_samples_split" : [2, 4, 10, 12, 16],
             "n_estimators": [50, 100, 400, 700]}

gs = GridSearchCV(estimator=rf1,
                  param_grid=param_grid,
                  scoring='accuracy',
                  cv=3,
                  n_jobs=-1)

gs = gs.fit(train.iloc[:, 2:], train.iloc[:, 1])
```


```python
print(gs.best_score_)
print(gs.best_params_)
#print(gs.cv_results_)
```

    0.824915824916
    {'min_samples_split': 12, 'n_estimators': 50, 'criterion': 'gini', 'min_samples_leaf': 1}
    

#### Fit model


```python
rf2 = RandomForestClassifier(criterion='gini', 
                             n_estimators=50,
                             min_samples_split=12,
                             min_samples_leaf=1,
                             max_features='auto',
                             oob_score=True,
                             random_state=1,
                             n_jobs=-1)
rf2.fit(train.iloc[:, 2:], train.iloc[:, 1])
print "%.4f" % rf2.oob_score_ 
```

    0.8238
    

#### Obtain cross-validation score with optimal hyperparameters


```python
scores2 = cross_val_score(rf1, train.iloc[:, 2:], train.iloc[:, 1], n_jobs=-1)
scores2.mean()
```




    0.80359147025813693



#### Inspect feature ranking


```python
pd.concat((pd.DataFrame(train.iloc[:, 2:].columns, columns = ['variable']), 
           pd.DataFrame(rf2.feature_importances_, columns = ['importance'])), 
          axis = 1).sort_values(by='importance', ascending = False)[:20]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>variable</th>
      <th>importance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>20</th>
      <td>Name_Title_Mr.</td>
      <td>0.116731</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Sex_male</td>
      <td>0.115737</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Sex_female</td>
      <td>0.112738</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Name_Len</td>
      <td>0.105312</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Fare</td>
      <td>0.103418</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Age</td>
      <td>0.084431</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Pclass_3</td>
      <td>0.062422</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Name_Title_Miss.</td>
      <td>0.035632</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Name_Title_Mrs.</td>
      <td>0.032156</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Pclass_1</td>
      <td>0.024577</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Cabin_Letter_n</td>
      <td>0.022817</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Pclass_2</td>
      <td>0.021992</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Embarked_C</td>
      <td>0.013963</td>
    </tr>
    <tr>
      <th>28</th>
      <td>SibSp_1</td>
      <td>0.012762</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Embarked_S</td>
      <td>0.011804</td>
    </tr>
    <tr>
      <th>29</th>
      <td>SibSp_0</td>
      <td>0.011327</td>
    </tr>
    <tr>
      <th>35</th>
      <td>Parch_0</td>
      <td>0.011118</td>
    </tr>
    <tr>
      <th>37</th>
      <td>Parch_2</td>
      <td>0.009834</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Age_Null_Flag</td>
      <td>0.009783</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Cabin_Letter_D</td>
      <td>0.009605</td>
    </tr>
  </tbody>
</table>
</div>



#### Generate submission file


```python
test['Fare'].fillna(train['Fare'].mean(), inplace = True)
predictions = rf2.predict(test.iloc[:, 1:])
predictions = pd.DataFrame(predictions, columns=['Survived'])
predictions = pd.concat((test.iloc[:, 0], predictions), axis = 1)
predictions.to_csv(os.path.join('submission_files', 'y_test1.csv'), sep=",", index = False)
```


```python
train.columns
```




    Index([u'PassengerId', u'Survived', u'Age', u'Fare', u'Name_Len',
           u'Age_Null_Flag', u'Pclass_3', u'Pclass_1', u'Pclass_2', u'Sex_male',
           u'Sex_female', u'Embarked_S', u'Embarked_C', u'Embarked_Q',
           u'Cabin_Letter_n', u'Cabin_Letter_C', u'Cabin_Letter_E',
           u'Cabin_Letter_G', u'Cabin_Letter_D', u'Cabin_Letter_A',
           u'Cabin_Letter_B', u'Cabin_Letter_F', u'Name_Title_Mr.',
           u'Name_Title_Mrs.', u'Name_Title_Miss.', u'Name_Title_Master.',
           u'Name_Title_Rev.', u'Name_Title_Dr.', u'Name_Title_Ms.',
           u'Name_Title_Col.', u'SibSp_1', u'SibSp_0', u'SibSp_3', u'SibSp_4',
           u'SibSp_2', u'SibSp_5', u'SibSp_8', u'Parch_0', u'Parch_1', u'Parch_2',
           u'Parch_5', u'Parch_3', u'Parch_4', u'Parch_6'],
          dtype='object')



# Model 3


```python
train = pd.read_csv(os.path.join('data', 'train.csv'))
test = pd.read_csv(os.path.join('data', 'test.csv'))
```


```python
train, test = feature_process_helper.names(train, test)
train, test = feature_process_helper.age_impute(train, test)
train, test = feature_process_helper.cabin(train, test)
train, test = feature_process_helper.embarked_impute(train, test)
train, test = feature_process_helper.dummies(train, test)
train, test = feature_process_helper.drop(train, test)
```


```python
y_train = train.iloc[:, 1]
```


```python
del train['Survived']
```


```python
rf3 = RandomForestClassifier(max_features='auto',
                                oob_score=True,
                                random_state=1,
                                n_jobs=-1)
```


```python
param_grid = { "criterion"   : ["gini", "entropy"],
             "min_samples_leaf" : [1,5,10],
             "min_samples_split" : [2, 4, 10, 12, 16],
             "n_estimators": [50, 100, 400, 700]}

gs = GridSearchCV(estimator=rf3,
                  param_grid=param_grid,
                  scoring='accuracy',
                  cv=3,
                  n_jobs=-1)

gs = gs.fit(train, y_train)
```


```python
print(gs.best_score_)
print(gs.best_params_)
#print(gs.cv_results_)
```

    0.83164983165
    {'min_samples_split': 10, 'n_estimators': 400, 'criterion': 'entropy', 'min_samples_leaf': 1}
    


```python
rf3 = RandomForestClassifier(min_samples_split=10, 
                             n_estimators=400, 
                             criterion='entropy', 
                             min_samples_leaf=1,
                             max_features='auto',
                             oob_score=True,
                             random_state=1,
                             n_jobs=-1)
rf3.fit(train, y_train)
print "%.4f" % rf3.oob_score_ 
```

    0.8249
    


```python
scores3 = cross_val_score(rf3, train, y_train, n_jobs=-1)
scores3.mean()
```




    0.83164983164983164




```python
pd.concat((pd.DataFrame(train.iloc[:, 2:].columns, columns = ['variable']), 
           pd.DataFrame(rf3.feature_importances_, columns = ['importance'])), 
          axis = 1).sort_values(by='importance', ascending = False)[:20]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>variable</th>
      <th>importance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>Age_Null_Flag</td>
      <td>0.133448</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Pclass_3</td>
      <td>0.107066</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Fare</td>
      <td>0.103650</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Name_Len</td>
      <td>0.103143</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Name_Title_Miss.</td>
      <td>0.099552</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Embarked_C</td>
      <td>0.092503</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Embarked_S</td>
      <td>0.087784</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Pclass_2</td>
      <td>0.052513</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Cabin_Letter_E</td>
      <td>0.033119</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Sex_male</td>
      <td>0.026176</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Name_Title_Rev.</td>
      <td>0.025992</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Sex_female</td>
      <td>0.021418</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Name_Title_Master.</td>
      <td>0.020970</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Embarked_Q</td>
      <td>0.016335</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Pclass_1</td>
      <td>0.012140</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Cabin_Letter_n</td>
      <td>0.011433</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Name_Title_Dr.</td>
      <td>0.011252</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Cabin_Letter_C</td>
      <td>0.008559</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Cabin_Letter_D</td>
      <td>0.008033</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Name_Title_Mr.</td>
      <td>0.005116</td>
    </tr>
  </tbody>
</table>
</div>




```python
test['Fare'].fillna(train['Fare'].mean(), inplace = True)
predictions = rf3.predict(test)
predictions = pd.DataFrame(predictions, columns=['Survived'])
predictions = pd.concat((test.iloc[:, 0], predictions), axis = 1)
predictions.to_csv(os.path.join('submission_files', 'y_test3.csv'), sep=",", index = False)
```

# RF Model 3


```python
train = pd.read_csv(os.path.join('data', 'train.csv'))
test = pd.read_csv(os.path.join('data', 'test.csv'))
train, test = feature_process_helper.names(train, test)
train, test = feature_process_helper.age_impute(train, test)
train, test = feature_process_helper.cabin(train, test)
train, test = feature_process_helper.embarked_impute(train, test)
train, test = feature_process_helper.fam_size(train, test)
train, test = feature_process_helper.dummies(train, test, columns = ['Pclass', 'Sex', 'Embarked',
                                                                     'Cabin_Letter', 'Name_Title', 'Fam_Size'])
train, test = feature_process_helper.drop(train, test)
```


```python
rf1 = RandomForestClassifier(max_features='auto',
                                oob_score=True,
                                random_state=1,
                                n_jobs=-1)

param_grid = { "criterion"   : ["gini", "entropy"],
             "min_samples_leaf" : [1,5,10],
             "min_samples_split" : [2, 4, 10, 12, 16],
             "n_estimators": [50, 100, 400, 700]}

gs = GridSearchCV(estimator=rf1,
                  param_grid=param_grid,
                  scoring='accuracy',
                  cv=3,
                  n_jobs=-1)

gs = gs.fit(train.iloc[:, 2:], train.iloc[:, 1])
```


```python
print(gs.best_score_)
print(gs.best_params_)
#print(gs.cv_results_)
```

    0.827160493827
    {'min_samples_split': 12, 'n_estimators': 50, 'criterion': 'gini', 'min_samples_leaf': 5}
    


```python
rf3 = RandomForestClassifier(criterion='gini', 
                             n_estimators=50,
                             min_samples_split=12,
                             min_samples_leaf=5,
                             max_features='auto',
                             oob_score=True,
                             random_state=1,
                             n_jobs=-1)
rf3.fit(train.iloc[:, 2:], train.iloc[:, 1])
print "%.4f" % rf3.oob_score_ 
```

    0.8406
    


```python
scores3 = cross_val_score(rf3, train.iloc[:, 2:], train.iloc[:, 1], n_jobs=-1)
scores3.mean()
```




    0.8271604938271605




```python
pd.concat((pd.DataFrame(train.iloc[:, 2:].columns, columns = ['variable']), 
           pd.DataFrame(rf3.feature_importances_, columns = ['importance'])), 
          axis = 1).sort_values(by='importance', ascending = False)[:20]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>variable</th>
      <th>importance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>7</th>
      <td>Sex_male</td>
      <td>0.190479</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Name_Title_Mr.</td>
      <td>0.147180</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Fare</td>
      <td>0.099675</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Sex_female</td>
      <td>0.091025</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Pclass_3</td>
      <td>0.071663</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Age</td>
      <td>0.071243</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Name_Len</td>
      <td>0.069634</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Cabin_Letter_n</td>
      <td>0.043399</td>
    </tr>
    <tr>
      <th>30</th>
      <td>Fam_Size_Big</td>
      <td>0.034171</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Name_Title_Mrs.</td>
      <td>0.030305</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Name_Title_Miss.</td>
      <td>0.024712</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Fam_Size_Nuclear</td>
      <td>0.020109</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Pclass_1</td>
      <td>0.016317</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Pclass_2</td>
      <td>0.014729</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Name_Title_Master.</td>
      <td>0.013780</td>
    </tr>
    <tr>
      <th>29</th>
      <td>Fam_Size_Solo</td>
      <td>0.012797</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Embarked_S</td>
      <td>0.010712</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Embarked_C</td>
      <td>0.008754</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Cabin_Letter_E</td>
      <td>0.006114</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Embarked_Q</td>
      <td>0.006108</td>
    </tr>
  </tbody>
</table>
</div>




```python
test['Fare'].fillna(train['Fare'].mean(), inplace = True)
predictions = rf3.predict(test.iloc[:, 1:])
predictions = pd.DataFrame(predictions, columns=['Survived'])
predictions = pd.concat((test.iloc[:, 0], predictions), axis = 1)
predictions.to_csv(os.path.join('submission_files', 'y_test3.csv'), sep=",", index = False)
```

Leaderboard score: 0.80383  
1024 out of 6067 (top 17%)

# Model 4



```python
train = pd.read_csv(os.path.join('data', 'train.csv'))
test = pd.read_csv(os.path.join('data', 'test.csv'))
test['Fare'].fillna(train['Fare'].mean(), inplace = True)
train, test = feature_process_helper.names(train, test)
train, test = feature_process_helper.age_impute(train, test)
train, test = feature_process_helper.cabin(train, test)
train, test = feature_process_helper.embarked_impute(train, test)
train, test = feature_process_helper.fam_size(train, test)
train, test = feature_process_helper.dummies(train, test, columns = ['Pclass', 'Sex', 'Embarked',
                                                                     'Cabin_Letter', 'Name_Title', 'Fam_Size'])
train, test = feature_process_helper.drop(train, test)
train, test = feature_process_helper.lda(train, test, train.iloc[:, 1], cols=['Age', 'Fare'])
```


```python
rf4 = RandomForestClassifier(max_features='auto',
                                oob_score=True,
                                random_state=1,
                                n_jobs=-1)

param_grid = { "criterion"   : ["gini", "entropy"],
             "min_samples_leaf" : [1,5,10],
             "min_samples_split" : [2, 4, 10, 12, 16],
             "n_estimators": [50, 100, 400, 700]}

gs = GridSearchCV(estimator=rf4,
                  param_grid=param_grid,
                  scoring='accuracy',
                  cv=3,
                  n_jobs=-1)

gs = gs.fit(train.iloc[:, 2:], train.iloc[:, 1])
```


```python
print(gs.best_score_)
print(gs.best_params_)
#print(gs.cv_results_)
```

    0.828282828283
    {'min_samples_split': 10, 'n_estimators': 50, 'criterion': 'entropy', 'min_samples_leaf': 1}
    


```python
rf4 = RandomForestClassifier(criterion='entropy', 
                             n_estimators=50,
                             min_samples_split=10,
                             min_samples_leaf=1,
                             max_features='auto',
                             oob_score=True,
                             random_state=1,
                             n_jobs=-1)
rf4.fit(train.iloc[:, 2:], train.iloc[:, 1])
print "%.4f" % rf4.oob_score_
```

    0.8126
    


```python
scores4 = cross_val_score(rf4, train.iloc[:, 2:], train.iloc[:, 1], n_jobs=-1)
scores4.mean()
```




    0.82828282828282829




```python
pd.concat((pd.DataFrame(train.iloc[:, 2:].columns, columns = ['variable']), 
           pd.DataFrame(rf4.feature_importances_, columns = ['importance'])), 
          axis = 1).sort_values(by='importance', ascending = False)[:20]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>variable</th>
      <th>importance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>29</th>
      <td>0</td>
      <td>0.194866</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Name_Len</td>
      <td>0.133324</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Sex_female</td>
      <td>0.116406</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Name_Title_Mr.</td>
      <td>0.094739</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Sex_male</td>
      <td>0.082101</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Pclass_3</td>
      <td>0.064193</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Cabin_Letter_n</td>
      <td>0.035912</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Fam_Size_Big</td>
      <td>0.033476</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Name_Title_Mrs.</td>
      <td>0.030457</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Pclass_1</td>
      <td>0.024086</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Name_Title_Miss.</td>
      <td>0.023415</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Fam_Size_Nuclear</td>
      <td>0.020619</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Pclass_2</td>
      <td>0.019736</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Name_Title_Master.</td>
      <td>0.019036</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Age_Null_Flag</td>
      <td>0.016703</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Embarked_S</td>
      <td>0.016258</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Fam_Size_Solo</td>
      <td>0.014073</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Embarked_C</td>
      <td>0.011576</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Embarked_Q</td>
      <td>0.008693</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Cabin_Letter_B</td>
      <td>0.008144</td>
    </tr>
  </tbody>
</table>
</div>




```python
predictions = rf4.predict(test.iloc[:, 1:])
predictions = pd.DataFrame(predictions, columns=['Survived'])
predictions = pd.concat((test.iloc[:, 0], predictions), axis = 1)
predictions.to_csv(os.path.join('submission_files', 'y_test4.csv'), sep=",", index = False)
```

Leaderboard score: 0.78

# Model 5


```python
train = pd.read_csv(os.path.join('data', 'train.csv'))
test = pd.read_csv(os.path.join('data', 'test.csv'))
test['Fare'].fillna(train['Fare'].mean(), inplace = True)
train, test = feature_process_helper.names(train, test)
train, test = feature_process_helper.age_impute(train, test)
train, test = feature_process_helper.cabin(train, test)
train, test = feature_process_helper.embarked_impute(train, test)
train, test = feature_process_helper.fam_size(train, test)
train, test = feature_process_helper.titles_grouped(train, test)
train, test = feature_process_helper.dummies(train, test, columns = ['Pclass', 'Sex', 'Embarked',
                                                                     'Cabin_Letter', 'Name_Title', 'Fam_Size'])
train, test = feature_process_helper.drop(train, test)
```


```python
rf5 = RandomForestClassifier(max_features='auto',
                                oob_score=True,
                                random_state=1,
                                n_jobs=-1)

param_grid = { "criterion"   : ["gini", "entropy"],
             "min_samples_leaf" : [1,5,10],
             "min_samples_split" : [2, 4, 10, 12, 16],
             "n_estimators": [50, 100, 400, 700]}

gs = GridSearchCV(estimator=rf5,
                  param_grid=param_grid,
                  scoring='accuracy',
                  cv=3,
                  n_jobs=-1)

gs = gs.fit(train.iloc[:, 2:], train.iloc[:, 1])
```


```python
print(gs.best_score_)
print(gs.best_params_)
#print(gs.cv_results_)
```

    0.829405162738
    {'min_samples_split': 12, 'n_estimators': 50, 'criterion': 'entropy', 'min_samples_leaf': 1}
    


```python
rf5 = RandomForestClassifier(criterion='entropy', 
                             n_estimators=50,
                             min_samples_split=12,
                             min_samples_leaf=1,
                             max_features='auto',
                             oob_score=True,
                             random_state=1,
                             n_jobs=-1)
rf5.fit(train.iloc[:, 2:], train.iloc[:, 1])
print "%.4f" % rf5.oob_score_
```

    0.8260
    


```python
scores5 = cross_val_score(rf5, train.iloc[:, 2:], train.iloc[:, 1], n_jobs=-1)
scores5.mean()
```




    0.82940516273849607




```python
pd.concat((pd.DataFrame(train.iloc[:, 2:].columns, columns = ['variable']), 
           pd.DataFrame(rf5.feature_importances_, columns = ['importance'])), 
          axis = 1).sort_values(by='importance', ascending = False)[:20]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>variable</th>
      <th>importance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Fare</td>
      <td>0.124604</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Age</td>
      <td>0.121088</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Name_Len</td>
      <td>0.106165</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Name_Title_Mr.</td>
      <td>0.099116</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Sex_male</td>
      <td>0.096142</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Sex_female</td>
      <td>0.083227</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Pclass_3</td>
      <td>0.051289</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Name_Title_Mrs.</td>
      <td>0.035439</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Name_Title_Miss.</td>
      <td>0.031467</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Pclass_1</td>
      <td>0.029258</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Cabin_Letter_n</td>
      <td>0.028717</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Fam_Size_Big</td>
      <td>0.027915</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Fam_Size_Nuclear</td>
      <td>0.026694</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Pclass_2</td>
      <td>0.018657</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Name_Title_Master.</td>
      <td>0.018434</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Age_Null_Flag</td>
      <td>0.014179</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Fam_Size_Solo</td>
      <td>0.013200</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Embarked_S</td>
      <td>0.013173</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Embarked_C</td>
      <td>0.010003</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Embarked_Q</td>
      <td>0.008866</td>
    </tr>
  </tbody>
</table>
</div>




```python
predictions = rf5.predict(test.iloc[:, 1:])
predictions = pd.DataFrame(predictions, columns=['Survived'])
predictions = pd.concat((test.iloc[:, 0], predictions), axis = 1)
predictions.to_csv(os.path.join('submission_files', 'y_test5.csv'), sep=",", index = False)
```

Leaderboard score: 0.79904

# RF 6


```python
train = pd.read_csv(os.path.join('data', 'train.csv'))
test = pd.read_csv(os.path.join('data', 'test.csv'))
train, test = feature_process_helper.names(train, test)
train, test = feature_process_helper.age_impute(train, test)
train, test = feature_process_helper.cabin(train, test)
train, test = feature_process_helper.embarked_impute(train, test)
train, test = feature_process_helper.fam_size(train, test)
train, test = feature_process_helper.dummies(train, test, columns = ['Pclass', 'Sex', 'Embarked',
                                                                     'Cabin_Letter', 'Name_Title', 'Fam_Size'])
train, test = feature_process_helper.drop(train, test)
```


```python
rf3 = RandomForestClassifier(criterion='gini', 
                             n_estimators=300,
                             min_samples_split=12,
                             min_samples_leaf=5,
                             max_features='auto',
                             oob_score=True,
                             random_state=1,
                             n_jobs=-1)
rf3.fit(train.iloc[:, 2:], train.iloc[:, 1])
print "%.4f" % rf3.oob_score_ 
```

    0.8305
    


```python
test['Fare'].fillna(train['Fare'].mean(), inplace = True)
predictions = rf3.predict(test.iloc[:, 1:])
predictions = pd.DataFrame(predictions, columns=['Survived'])
predictions = pd.concat((test.iloc[:, 0], predictions), axis = 1)
predictions.to_csv(os.path.join('submission_files', 'y_test6.csv'), sep=",", index = False)
```

Leaderboard score: 0.80861  
563 out of 6067 (top 9.3%)

# RF 7


```python
train = pd.read_csv(os.path.join('data', 'train.csv'))
test = pd.read_csv(os.path.join('data', 'test.csv'))
train, test = feature_process_helper.names(train, test)
train, test = feature_process_helper.age_impute(train, test)
train, test = feature_process_helper.cabin(train, test)
train, test = feature_process_helper.embarked_impute(train, test)
train, test = feature_process_helper.fam_size(train, test)
train, test = feature_process_helper.ticket_grouped(train, test)
train, test = feature_process_helper.dummies(train, test, columns = ['Pclass', 'Sex', 'Embarked', 'Ticket_Lett',
                                                                     'Cabin_Letter', 'Name_Title', 'Fam_Size'])
train, test = feature_process_helper.drop(train, test)
```


```python
rf7 = RandomForestClassifier(max_features='auto',
                                oob_score=True,
                                random_state=1,
                                n_jobs=-1)

param_grid = { "criterion"   : ["gini", "entropy"],
             "min_samples_leaf" : [1,5,10],
             "min_samples_split" : [2, 4, 10, 12, 16],
             "n_estimators": [50, 100, 400, 700]}

gs = GridSearchCV(estimator=rf7,
                  param_grid=param_grid,
                  scoring='accuracy',
                  cv=3,
                  n_jobs=-1)

gs = gs.fit(train.iloc[:, 2:], train.iloc[:, 1])
```


```python
print(gs.best_score_)
print(gs.best_params_)
#print(gs.cv_results_)
```

    0.838383838384
    {'min_samples_split': 4, 'n_estimators': 100, 'criterion': 'gini', 'min_samples_leaf': 1}
    


```python
rf7 = RandomForestClassifier(criterion='gini', 
                             n_estimators=100,
                             min_samples_split=4,
                             min_samples_leaf=1,
                             max_features='auto',
                             oob_score=True,
                             random_state=1,
                             n_jobs=-1)
rf7.fit(train.iloc[:, 2:], train.iloc[:, 1])
print "%.4f" % rf7.oob_score_
```

    0.8283
    


```python
scores7 = cross_val_score(rf7, train.iloc[:, 2:], train.iloc[:, 1], n_jobs=-1)
scores7.mean()
```




    0.83838383838383834




```python
pd.concat((pd.DataFrame(train.iloc[:, 2:].columns, columns = ['variable']), 
           pd.DataFrame(rf7.feature_importances_, columns = ['importance'])), 
          axis = 1).sort_values(by='importance', ascending = False)[:20]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>variable</th>
      <th>importance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Fare</td>
      <td>0.120061</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Name_Len</td>
      <td>0.117291</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Age</td>
      <td>0.116984</td>
    </tr>
    <tr>
      <th>29</th>
      <td>Name_Title_Mr.</td>
      <td>0.111207</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Sex_female</td>
      <td>0.082661</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Sex_male</td>
      <td>0.074994</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Pclass_3</td>
      <td>0.039930</td>
    </tr>
    <tr>
      <th>39</th>
      <td>Fam_Size_Big</td>
      <td>0.028404</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Cabin_Letter_n</td>
      <td>0.028148</td>
    </tr>
    <tr>
      <th>31</th>
      <td>Name_Title_Miss.</td>
      <td>0.027901</td>
    </tr>
    <tr>
      <th>30</th>
      <td>Name_Title_Mrs.</td>
      <td>0.026725</td>
    </tr>
    <tr>
      <th>37</th>
      <td>Fam_Size_Nuclear</td>
      <td>0.020204</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Pclass_1</td>
      <td>0.019251</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Ticket_Lett_1</td>
      <td>0.017591</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Ticket_Lett_3</td>
      <td>0.013626</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Ticket_Lett_Low_ticket</td>
      <td>0.013341</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Embarked_S</td>
      <td>0.012750</td>
    </tr>
    <tr>
      <th>38</th>
      <td>Fam_Size_Solo</td>
      <td>0.012585</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Age_Null_Flag</td>
      <td>0.011939</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Pclass_2</td>
      <td>0.011832</td>
    </tr>
  </tbody>
</table>
</div>




```python
test['Fare'].fillna(train['Fare'].mean(), inplace = True)
predictions = rf7.predict(test.iloc[:, 1:])
predictions = pd.DataFrame(predictions, columns=['Survived'])
predictions = pd.concat((test.iloc[:, 0], predictions), axis = 1)
predictions.to_csv(os.path.join('submission_files', 'y_test7.csv'), sep=",", index = False)
```

Leaderboard score: 0.7894

# RF 8

I tried RF 7 with 500 estimators, and the score went down to .7799

# RF 9


```python
train = pd.read_csv(os.path.join('data', 'train.csv'))
test = pd.read_csv(os.path.join('data', 'test.csv'))
train, test = feature_process_helper.names(train, test)
train, test = feature_process_helper.age_impute(train, test)
train, test = feature_process_helper.cabin(train, test)
train, test = feature_process_helper.embarked_impute(train, test)
train, test = feature_process_helper.fam_size(train, test)
train, test = feature_process_helper.dummies(train, test, columns = ['Pclass', 'Sex', 'Embarked', 
                                                                     'Cabin_Letter', 'Name_Title', 'Fam_Size'])
train['Ticket_Len'] = train['Ticket'].apply(lambda x: len(x))
test['Ticket_Len'] = test['Ticket'].apply(lambda x: len(x))
train, test = feature_process_helper.drop(train, test)
```


```python
rf9 = RandomForestClassifier(max_features='auto',
                                oob_score=True,
                                random_state=1,
                                n_jobs=-1)

param_grid = { "criterion"   : ["gini", "entropy"],
             "min_samples_leaf" : [1,5,10],
             "min_samples_split" : [2, 4, 10, 12, 16],
             "n_estimators": [50, 100, 400, 700]}

gs = GridSearchCV(estimator=rf9,
                  param_grid=param_grid,
                  scoring='accuracy',
                  cv=3,
                  n_jobs=-1)

gs = gs.fit(train.iloc[:, 2:], train.iloc[:, 1])
```


```python
print(gs.best_score_)
print(gs.best_params_)
#print(gs.cv_results_)
```

    0.836139169473
    {'min_samples_split': 4, 'n_estimators': 50, 'criterion': 'entropy', 'min_samples_leaf': 1}
    


```python
rf9 = RandomForestClassifier(criterion='entropy', 
                             n_estimators=50,
                             min_samples_split=4,
                             min_samples_leaf=1,
                             max_features='auto',
                             oob_score=True,
                             random_state=1,
                             n_jobs=-1)
rf9.fit(train.iloc[:, 2:], train.iloc[:, 1])
print "%.4f" % rf9.oob_score_
```

    0.8227
    


```python
scores9 = cross_val_score(rf9, train.iloc[:, 2:], train.iloc[:, 1], n_jobs=-1)
scores9.mean()
```




    0.83613916947250277




```python
pd.concat((pd.DataFrame(train.iloc[:, 2:].columns, columns = ['variable']), 
           pd.DataFrame(rf9.feature_importances_, columns = ['importance'])), 
          axis = 1).sort_values(by='importance', ascending = False)[:20]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>variable</th>
      <th>importance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>Name_Len</td>
      <td>0.136529</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Fare</td>
      <td>0.135527</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Age</td>
      <td>0.132001</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Sex_male</td>
      <td>0.083224</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Sex_female</td>
      <td>0.073391</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Name_Title_Mr.</td>
      <td>0.070640</td>
    </tr>
    <tr>
      <th>31</th>
      <td>Ticket_Len</td>
      <td>0.063579</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Pclass_3</td>
      <td>0.040833</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Cabin_Letter_n</td>
      <td>0.026080</td>
    </tr>
    <tr>
      <th>30</th>
      <td>Fam_Size_Big</td>
      <td>0.025057</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Name_Title_Miss.</td>
      <td>0.022915</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Name_Title_Mrs.</td>
      <td>0.021901</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Fam_Size_Nuclear</td>
      <td>0.021485</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Pclass_1</td>
      <td>0.019593</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Pclass_2</td>
      <td>0.017138</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Age_Null_Flag</td>
      <td>0.014891</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Embarked_S</td>
      <td>0.013172</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Name_Title_Master.</td>
      <td>0.012711</td>
    </tr>
    <tr>
      <th>29</th>
      <td>Fam_Size_Solo</td>
      <td>0.011858</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Embarked_C</td>
      <td>0.011548</td>
    </tr>
  </tbody>
</table>
</div>




```python
test['Fare'].fillna(train['Fare'].mean(), inplace = True)
predictions = rf9.predict(test.iloc[:, 1:])
predictions = pd.DataFrame(predictions, columns=['Survived'])
predictions = pd.concat((test.iloc[:, 0], predictions), axis = 1)
predictions.to_csv(os.path.join('submission_files', 'y_test9.csv'), sep=",", index = False)
```

Leaderboard score: 0.78469

# Model 10


```python
rf10 = RandomForestClassifier(criterion='gini', 
                             n_estimators=300,
                             min_samples_split=12,
                             min_samples_leaf=5,
                             max_features='auto',
                             oob_score=True,
                             random_state=1,
                             n_jobs=-1)
rf10.fit(train.iloc[:, 2:], train.iloc[:, 1])
print "%.4f" % rf10.oob_score_ 
```

    0.8316
    


```python
test['Fare'].fillna(train['Fare'].mean(), inplace = True)
predictions = rf10.predict(test.iloc[:, 1:])
predictions = pd.DataFrame(predictions, columns=['Survived'])
predictions = pd.concat((test.iloc[:, 0], predictions), axis = 1)
predictions.to_csv(os.path.join('submission_files', 'y_test10.csv'), sep=",", index = False)
```

Leaderborad score: 0.81340  
445 out of 6069 (top 8%)

# Model 11


```python
train = pd.read_csv(os.path.join('data', 'train.csv'))
test = pd.read_csv(os.path.join('data', 'test.csv'))
train, test = feature_process_helper.names(train, test)
train, test = feature_process_helper.age_impute(train, test)
train, test = feature_process_helper.cabin(train, test)
train, test = feature_process_helper.embarked_impute(train, test)
train, test = feature_process_helper.fam_size(train, test)
train, test = feature_process_helper.ticket_grouped(train, test)
train, test = feature_process_helper.dummies(train, test, columns = ['Pclass', 'Sex', 'Embarked', 'Ticket_Lett',
                                                                     'Cabin_Letter', 'Name_Title', 'Fam_Size'])
train, test = feature_process_helper.drop(train, test)
```


```python
rf11 = RandomForestClassifier(criterion='gini', 
                             n_estimators=300,
                             min_samples_split=12,
                             min_samples_leaf=5,
                             max_features='auto',
                             oob_score=True,
                             random_state=1,
                             n_jobs=-1)
rf11.fit(train.iloc[:, 2:], train.iloc[:, 1])
print "%.4f" % rf11.oob_score_ 
```

    0.8283
    


```python
test['Fare'].fillna(train['Fare'].mean(), inplace = True)
predictions = rf11.predict(test.iloc[:, 1:])
predictions = pd.DataFrame(predictions, columns=['Survived'])
predictions = pd.concat((test.iloc[:, 0], predictions), axis = 1)
predictions.to_csv(os.path.join('submission_files', 'y_test11.csv'), sep=",", index = False)
```

Leaderboard score: 0.81340  
445 out of 6069 (top 8%)

# Model 12


```python
rf12 = RandomForestClassifier(criterion='gini', 
                             n_estimators=1000,
                             min_samples_split=12,
                             min_samples_leaf=5,
                             max_features='auto',
                             oob_score=True,
                             random_state=1,
                             n_jobs=-1)
rf12.fit(train.iloc[:, 2:], train.iloc[:, 1])
print "%.4f" % rf12.oob_score_ 
```

    0.8350
    


```python
test['Fare'].fillna(train['Fare'].mean(), inplace = True)
predictions = rf12.predict(test.iloc[:, 1:])
predictions = pd.DataFrame(predictions, columns=['Survived'])
predictions = pd.concat((test.iloc[:, 0], predictions), axis = 1)
predictions.to_csv(os.path.join('submission_files', 'y_test12.csv'), sep=",", index = False)
```

Leaderboard score: 
