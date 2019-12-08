#!/usr/bin/python
# coding: utf-8
'''
Created on 2018-05-14
Update  on 2018-05-14
Author: 平淡的天
Github: https://github.com/apachecn/kaggle
'''

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import pandas as pd

train_data = pd.read_csv(r"C:\Users\312\Desktop\digit-recognizer\train.csv")
test_data = pd.read_csv(r"C:\Users\312\Desktop\digit-recognizer\test.csv")
data = pd.concat([train_data, test_data], axis=0).reset_index(drop=True)
data.drop(['label'], axis=1, inplace=True)
label = train_data.label

pca = PCA(n_components=100, random_state=34)
data_pca = pca.fit_transform(data)

Xtrain, Ytrain, xtest, ytest = train_test_split(
    data_pca[0:len(train_data)], label, test_size=0.1, random_state=34)

clf = MLPClassifier(
    hidden_layer_sizes=(100, ),
    activation='relu',
    alpha=0.0001,
    learning_rate='constant',
    learning_rate_init=0.001,
    max_iter=200,
    shuffle=True,
    random_state=34)

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
