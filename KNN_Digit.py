#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# @Time : 2019/2/6 15:10 
# @Author : Patrick 
# @File : KNN_Digit.py 
# @Software: PyCharm


import time

import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

train_pect = 0.8
k = 3

data_set = pd.read_csv('Digit Recognizer/train.csv')
print(data_set)
data = data_set.drop(labels='label', axis=1)
print(data)
print(data.size)
print('------------------------------------------------')
data = preprocessing.minmax_scale(data)
label = data_set['label']
print(data.size)
print(label)

X_train, X_test, y_train, y_test = train_test_split(data, label, random_state=1, train_size=train_pect,
                                                    test_size=1 - train_pect)
#
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
#
t1 = time.time()
classifier = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train.values.ravel())
t2 = time.time()
print('Training time: ' + str(t2 - t1) + 's')

predicted = classifier.predict(X_test)
t3 = time.time()
print('Predict time: ' + str(t3 - t2) + 's')

#
# # print(len(predicted))
matrix = [[0 for i in range(10)] for i in range(10)]

count = 0
for i in zip(predicted, y_test):
    real, pred = i
    if real == pred:
        count += 1
    matrix[pred][real] += 1
t4 = time.time()

print(count / len(y_test))
for i in matrix:
    print(i)
print('Compare time: ' + str(t4 - t3) + 's')
