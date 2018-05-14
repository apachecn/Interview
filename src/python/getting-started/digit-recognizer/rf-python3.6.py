#!/usr/bin/python
# coding: utf-8
'''
Created on 2018-05-14
Update  on 2018-05-14
Author: 平淡的天
Github: https://github.com/apachecn/kaggle
'''

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import pandas as pd
# from sklearn.grid_search import GridSearchCV
# from numpy import arange
# from lightgbm import LGBMClassifier

train_data = pd.read_csv(r"C:\Users\312\Desktop\digit-recognizer\train.csv")
test_data = pd.read_csv(r"C:\Users\312\Desktop\digit-recognizer\test.csv")
data = pd.concat([train_data, test_data], axis=0).reset_index(drop=True)
data.drop(['label'], axis=1, inplace=True)
label = train_data.label

pca = PCA(n_components=100, random_state=34)
data_pca = pca.fit_transform(data)

Xtrain, Ytrain, xtest, ytest = train_test_split(
    data_pca[0:len(train_data)], label, test_size=0.1, random_state=34)

clf = RandomForestClassifier(
    n_estimators=110,
    max_depth=5,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=34)

# clf=LGBMClassifier(num_leaves=63, max_depth=7, n_estimators=80, n_jobs=20)

# param_test1 = {'n_estimators':arange(10,150,10),'max_depth':arange(1,11,1)}
# gsearch1 = GridSearchCV(estimator = clf, param_grid = param_test1, scoring='accuracy',iid=False,cv=5)
# gsearch1.fit(Xtrain,xtest)
# print(gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_)

clf.fit(Xtrain, xtest)
y_predict = clf.predict(Ytrain)

zeroLable = ytest - y_predict
rightCount = 0
for i in range(len(zeroLable)):
    if list(zeroLable)[i] == 0:
        rightCount += 1
print('the right rate is:', float(rightCount) / len(zeroLable))

result = clf.predict(data_pca[len(train_data):])

i = 0
fw = open("C:\\Users\\312\\Desktop\\digit-recognizer\\result.csv", 'w')
with open('C:\\Users\\312\\Desktop\\digit-recognizer\\sample_submission.csv'
          ) as pred_file:
    fw.write('{},{}\n'.format('ImageId', 'Label'))
    for line in pred_file.readlines()[1:]:
        splits = line.strip().split(',')
        fw.write('{},{}\n'.format(splits[0], result[i]))
        i += 1
