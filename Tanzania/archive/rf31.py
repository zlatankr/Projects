import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import feature_process_helper

X_train = pd.read_csv('X_train.csv')
X_test = pd.read_csv('X_test.csv')
y_train = pd.read_csv('y_train.csv')
del y_train['id']
X_train, X_test = feature_process_helper.dates(X_train, X_test)
X_train, X_test = feature_process_helper.dates2(X_train, X_test)
X_train, X_test = feature_process_helper.construction(X_train, X_test)
X_train, X_test = feature_process_helper.bools(X_train, X_test)
X_train, X_test = feature_process_helper.locs(X_train, X_test)
X_train['population'] = np.log(X_train['population'])
X_test['population'] = np.log(X_test['population'])
X_train, X_test = feature_process_helper.removal(X_train, X_test)
X_train, X_test = feature_process_helper.lda(X_train, X_test, y_train, cols = ['gps_height', 'latitude', 'longitude'])
X_train, X_test = feature_process_helper.dummies(X_train, X_test)

rf31 = RandomForestClassifier(criterion='gini',
                                min_samples_split=8,
                                n_estimators=500,
                                max_features='auto',
                                oob_score=True,
                                random_state=1,
                                n_jobs=-1)
                            
rf31.fit(X_train, y_train.values.ravel())
print "%.4f" % rf31.oob_score_ 

if __name__ == '__main__':
    rf31()