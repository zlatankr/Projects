
"""

This files stores all the essential helper functions for variable pre-processing

"""
import numpy as np
import pandas as pd


def removal(X_train, X_test):
    """
    Here we define all the columns that we want to delete right off the bat.

    # id: we drop the id column because it is not a useful predictor.
    # 'amount_tsh' is mostly blank - delete
    # wpt_name: not useful, delete (too many values)
    # subvillage: too many values, delete
    # scheme_name: this is almost 50% nulls, so we will delete this column
    # num_private: we will delete this column because ~99% of the values are zeros.
    # region: drop this b/c is seems very similar to region_code, though not 100% sure about this one!
    """
    z = ['id', 'amount_tsh', 'num_private', 'wpt_name', 
          'subvillage', 'scheme_name', 'region', 'installer']
    for i in z:
        del X_train[i]
        del X_test[i]
    return X_train, X_test


def construction(X_train, X_test):
    """
    construction_year has 35% nulls, so we impute the nulls with the column mean
    """
    for i in [X_train, X_test]:
        i['construction_year'].replace(0, X_train[X_train['construction_year'] <> 0]['construction_year'].mean(), inplace=True)
    return X_train, X_test

def dates(X_train, X_test):
    """
    date_recorded: this might be a useful variable for this analysis, although the year itself would be useless in a practical scenario moving into the future. We will convert this column into a datetime, and we will also create 'year_recorded' and 'month_recorded' columns just in case those levels prove to be useful. A visual inspection of both casts significant doubt on that possibility, but we'll proceed for now. We will delete date_recorded itself, since random forest cannot accept datetime
    """
    for i in [X_train, X_test]:
        i['date_recorded'] = pd.to_datetime(i['date_recorded'])
        i['year_recorded'] = i['date_recorded'].apply(lambda x: x.year)
        i['month_recorded'] = i['date_recorded'].apply(lambda x: x.month)
        i['date_recorded'] = (pd.to_datetime(i['date_recorded'])).apply(lambda x: x.toordinal())
    return X_train, X_test

def locs(X_train, X_test):
    """
    gps_height, latitude, longitude, population
    """
    trans = ['longitude', 'latitude', 'gps_height', 'population']
    for i in [X_train, X_test]:
        i.loc[i.longitude == 0, 'latitude'] = 0
        for z in trans:
            i[z].replace(0., np.NaN, inplace = True)
            i[z].replace(1., np.NaN, inplace = True)
            data = X_train.groupby(['subvillage'])[z]
            i[z] = data.transform(lambda x: x.fillna(x.mean()))
            data = X_train.groupby(['district_code'])[z]
            i[z] = data.transform(lambda x: x.fillna(x.mean()))
            data = X_train.groupby(['basin'])[z]
            i[z] = data.transform(lambda x: x.fillna(x.mean()))
            i[z] = i[z].fillna(X_train[z].mean())
    return X_train, X_test

def bools(X_train, X_test):
    """
    public_meeting: we will fill the nulls as 'False'
    permit: we will fill the nulls as 'False
    """
    z = ['public_meeting', 'permit']
    for i in z:
        X_train[i].fillna(False, inplace = True)
        X_train[i] = X_train[i].apply(lambda x: float(x))
        X_test[i].fillna(False, inplace = True)
        X_test[i] = X_test[i].apply(lambda x: float(x))
    return X_train, X_test


def codes(X_train, X_test):
    """
    convert region_code and district_code to string objects, since they are actually categorical variables
    """
    for i in ['region_code', 'district_code']:
        X_train[i] = X_train[i].apply(lambda x: str(x))
        X_test[i] = X_test[i].apply(lambda x: str(x))
    return X_train, X_test

def dummies(X_train, X_test):
    columns = [i for i in X_train.columns if type(X_train[i].iloc[1]) == str]
    for column in columns:
        good_cols = []
        X_train[column].fillna('NULL', inplace = True)
        dumms = pd.get_dummies(X_train[column], prefix = column+'_')
        for i in dumms.columns:
        #    if chi2_contingency(pd.crosstab(dumms[i], y_train['status_group']))[1] < .001:
            good_cols.append(i)
        good_cols = [i for i in good_cols if i in pd.get_dummies(X_test[column], prefix = column+'_').columns]
        X_train = pd.concat((X_train, pd.get_dummies(X_train[column], prefix = column+'_')[good_cols]), axis = 1)
        X_test = pd.concat((X_test, pd.get_dummies(X_test[column], prefix = column+'_')[good_cols]), axis = 1)
        del X_train[column]
        del X_test[column]
    return X_train, X_test

def dummies2(X_train, X_test):
    columns = [i for i in X_train.columns if type(X_train[i].iloc[1]) == str]
    status = pd.get_dummies(y_train['status_group'])
    for column in columns:
        good_cols = []
        X_train[column].fillna('NULL', inplace = True)
        dumms = pd.get_dummies(X_train[column], prefix = column+'_')
        for i in dumms.columns:
            if status[dumms[i] == 1]['functional'].mean() > (status['functional'].mean() + .1):
                good_cols.append(i)
            elif status[dumms[i] == 1]['non functional'].mean() > (status['non functional'].mean() + .1):
                good_cols.append(i)
            elif status[dumms[i] == 1]['functional needs repair'].mean() > (status['functional needs repair'].mean() + .1):
                good_cols.append(i)
        good_cols = [i for i in good_cols if i in pd.get_dummies(X_test[column], prefix = column+'_').columns]
        X_train = pd.concat((X_train, pd.get_dummies(X_train[column], prefix = column+'_')[good_cols]), axis = 1)
        X_test = pd.concat((X_test, pd.get_dummies(X_test[column], prefix = column+'_')[good_cols]), axis = 1)
        del X_train[column]
        del X_test[column]
    return X_train, X_test