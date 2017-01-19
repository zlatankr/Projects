
#

# In[ ]:

corrs = joblib.load(os.path.join('pickles', 'rf1_corr_dict.pkl'))


# In[7]:

dump = [i.split('&')[1] for i in corrs.keys()]
keep = [i for i in X_train.columns if i not in dump]


# In[8]:

X_train = X_train[keep]
X_test = X_test[keep]


# In[10]:

[i for i in X_train.columns if i.split('_')[0] == 'amntsh']


# In[12]:

del X_train['amntshnull']
del X_train['amntsh_[0.2, 50]']
del X_train['amntsh_(50, 250]']
del X_train['amntsh_(250, 1000]']
del X_train['amntsh_(1000, 350000]']


# In[5]:

np.shape(X_train)


# In[5]:

rf = RandomForestClassifier(criterion='entropy',
                                n_estimators=100,
                                max_features='auto',
                                oob_score=True,
                                random_state=1,
                                n_jobs=-1)
                            
rf = rf.fit(X_train, y_train.values.ravel())
print "%.4f" % rf.oob_score_ 


# In[11]:

pd.concat((pd.DataFrame(X_train.columns, columns = ['variable']), 
           pd.DataFrame(rf.feature_importances_, columns = ['importance'])), 
          axis = 1).sort_values(by='importance', ascending = False)[:10]


# #### Save model to pickle

# In[ ]:

joblib.dump(rf, os.path.join('pickles', 'rf1.pkl'), 9)


# #### Get model cross validation score

# In[77]:

scores = cross_val_score(rf, X_train, y_train.values.ravel(), n_jobs=-1)
scores.mean()


# #### Save cross validation score to pickle

# In[78]:

joblib.dump(scores, os.path.join('pickles', 'rf1_cv.pkl'), 9)


# In[73]:

def meaningful():
    global X_train, X_test
    status = pd.get_dummies(y_train['status_group'])
    good_cols = []
    for i in X_train.columns[12:]:
        if status[X_train[i] == 1]['functional'].mean() > (status['functional'].mean() + .0510):
            good_cols.append(i)
        elif status[X_train[i] == 1]['functional needs repair'].mean() > (status['functional needs repair'].mean() + .0510):
            good_cols.append(i)
        elif status[X_train[i] == 1]['non functional'].mean() > (status['non functional'].mean() + .0510):
            good_cols.append(i)
    X_train2 = pd.concat((X_train.iloc[:, :12], X_train[good_cols]), axis = 1)
    X_test2 = pd.concat((X_test.iloc[:, :12], X_test[good_cols]), axis = 1)
    return X_train2, X_test2


# In[74]:

X_train2, X_test2 = meaningful()


# ### Random Forest  2

# In[14]:

X_train, X_test = feature_process_helper.dates(X_train, X_test)
X_train, X_test = feature_process_helper.construction(X_train, X_test)
X_train, X_test = feature_process_helper.bools(X_train, X_test)
X_train, X_test = feature_process_helper.locs(X_train, X_test)
X_train, X_test = feature_process_helper.removal(X_train, X_test)
X_train, X_test = feature_process_helper.dummies(X_train, X_test)


# In[76]:

corrs = joblib.load(os.path.join('pickles', 'rf1_corr_dict.pkl'))


# In[77]:

dump = [i.split('&')[1] for i in corrs.keys()]
keep = [i for i in X_train.columns if i not in dump]


# In[78]:

X_train2 = X_train2[keep]
X_test2 = X_test2[keep]


# In[79]:

len(X_train2.columns)


# In[ ]:

rf2 = RandomForestClassifier(criterion='entropy',
                                n_estimators=100,
                                max_features='auto',
                                oob_score=True,
                                random_state=1,
                                n_jobs=-1)
                            
rf2 = rf2.fit(X_train2, y_train.values.ravel())
print "%.4f" % rf2.oob_score_ 


# In[25]:

predictions = rf2.predict(X_test)
y_test = pd.read_csv('y_test.csv')
pred = pd.DataFrame(predictions, columns = [y_test.columns[1]])
del y_test['status_group']
y_test = pd.concat((y_test, pred), axis = 1)
y_test.to_csv('y_test1.csv', sep=",", index = False)


# In[123]:

smalls = []
for i in X_train.columns:
    if sum(X_train[i]) < 50:
        smalls.append(i)


# In[127]:

keep = [i for i in X_train.columns if i not in smalls]


# In[128]:

X_train = X_train[keep]
X_test = X_test[keep]

