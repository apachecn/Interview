#!/usr/bin/python
# coding: utf-8
'''
Created on 2018-05-14
Update  on 2018-05-14
Author: 平淡的天
Github: https://github.com/apachecn/kaggle
'''
import os.path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import pandas as pd
# from sklearn.grid_search import GridSearchCV
# from numpy import arange
# from lightgbm import LGBMClassifier
data_dir = \
    r'/Users/wuyanxue/Documents/GitHub/datasets/getting-started/digit-recognizer/'

train_data = pd.read_csv(os.path.join(data_dir, 'input/train.csv'))
test_data = pd.read_csv(os.path.join(data_dir, 'input/test.csv'))
data = pd.concat([train_data, test_data], axis=0).reset_index(drop=True)
data.drop(['label'], axis=1, inplace=True)
label = train_data.label

pca = PCA(n_components=100, random_state=34)
data_pca = pca.fit_transform(data)

X_train, X_test, y_train, y_test = train_test_split(
    data_pca[0:len(train_data)], label, test_size=0.1, random_state=34)

clf = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    min_samples_split=20,
    min_samples_leaf=1,
    random_state=34)
# clf=LGBMClassifier(num_leaves=63, max_depth=7, n_estimators=80, n_jobs=20)
# param_test1 = {'n_estimators':arange(10,150,10),'max_depth':arange(1,11,1)}
# gsearch1 = GridSearchCV(estimator = clf, param_grid = param_test1, scoring='accuracy',iid=False,cv=5)
# gsearch1.fit(Xtrain,xtest)
# print(gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_)

clf.fit(X_train, y_train)
y_predict = clf.predict(X_test)

zeroLable = y_test - y_predict
rightCount = 0
for i in range(len(zeroLable)):
    if list(zeroLable)[i] == 0:
        rightCount += 1
print('the right rate is:', float(rightCount) / len(zeroLable))

result = clf.predict(data_pca[len(train_data):])

n, _ = test_data.shape
with open(os.path.join(data_dir, 'output/Result_sklearn_RF.csv'), 'w') as fw:
    fw.write('{},{}\n'.format('ImageId', 'Label'))
    for i in range(1, n + 1):
        fw.write('{},{}\n'.format(i, result[i - 1]))
