
# coding: utf-8

# Download the training and testing datasets to our local directory.

# In[1]:

import requests
import csv

urls = {
        'X_train' : "https://s3.amazonaws.com/drivendata/data/7/public/4910797b-ee55-40a7-8668-10efd5c1b960.csv",
        'y_train' : "https://s3.amazonaws.com/drivendata/data/7/public/0bf8bc6e-30d0-4c50-956a-603fc693d966.csv",
        'X_test' : "https://s3.amazonaws.com/drivendata/data/7/public/702ddfc5-68cd-4d1d-a0de-f5f566f76d91.csv",
        'y_test' : "https://s3.amazonaws.com/drivendata/data/7/public/SubmissionFormat.csv"
        }

for i in urls:
    r = requests.get(urls[i])

    text = r.iter_lines()

    reader = csv.reader(text, delimiter=',')

    mylist = list(reader)

    with open(str(i)+'.csv', 'wb') as fp:
        a = csv.writer(fp, delimiter=',')
        data = mylist
        a.writerows(data)