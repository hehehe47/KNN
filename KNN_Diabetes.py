#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# @Time : 2019/2/2 17:09 
# @Author : Patrick 
# @File : KNN_Diabetes.py
# @Software: PyCharm


import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

label_used = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction',
              'Age']

train_pect = 0.8
k = 3

# test_data = pd.read_csv('Digit Recognizer/test.csv')

data_set = pd.read_csv('Diabetes/diabetes.csv')

data = data_set[label_used]
d = preprocessing.minmax_scale(data)
label = data_set[['Outcome']]
#
X_train, X_test, y_train, y_test = train_test_split(data, label, random_state=1, train_size=train_pect,
                                                    test_size=1 - train_pect)
#
# print(X_train.shape)
# print(y_train.shape)
# # print(X_test.shape)
# # print(y_test.shape)
#
classifier = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train.values.ravel())

predicted = classifier.predict(X_test[label_used])

# print(len(predicted))
count = 0
for i in zip(predicted, y_test['Outcome']):
    real, pred = i
    if real == pred:
        count += 1

print(count / len(y_test))
