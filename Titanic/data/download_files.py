
# coding: utf-8

# Download the training and testing datasets to our local directory.

import requests
import csv

urls = {
        'train' : "https://www.kaggle.com/c/titanic/download/train.csv",
        'test' : "https://www.kaggle.com/c/titanic/download/test.csv"
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