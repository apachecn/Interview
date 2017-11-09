#!/usr/bin/python
# coding: utf-8

'''
Created on 2017-10-26
Update  on 2017-10-26
@author: 片刻
Github:  https://github.com/apachecn/kaggle
'''

import csv
import time
import pandas as pd
from numpy import *
from sklearn.neighbors import KNeighborsClassifier  


# 加载数据
def opencsv():
    # 使用 pandas 打开
    data = pd.read_csv('datasets/input/GettingStarted/DigitRecognizer/train.csv')
    data1 = pd.read_csv('datasets/input/GettingStarted/DigitRecognizer/test.csv')

    train_data = data.values[0:, 1:]  # 读入全部训练数据
    train_label = data.values[0:, 0]
    test_data = data1.values[0:, 0:]  # 测试全部测试个数据
    return train_data, train_label, test_data


def saveResult(result, csvName):
    with open(csvName, 'wb') as myFile:
        myWriter = csv.writer(myFile)
        myWriter.writerow(["ImageId", "Label"])
        index = 0
        for i in result:
            tmp = []
            index = index+1
            tmp.append(index)
            # tmp.append(i)
            tmp.append(int(i))
            myWriter.writerow(tmp)


def knnClassify(trainData, trainLabel, testData): 
    knnClf = KNeighborsClassifier()   # default:k = 5,defined by yourself:KNeighborsClassifier(n_neighbors=10)
    knnClf.fit(trainData, ravel(trainLabel))
    testLabel = knnClf.predict(testData)
    saveResult(testLabel, 'datasets/ouput/GettingStarted/DigitRecognizer/Result_sklearn_knn.csv')
    return testLabel


def dRecognition_knn():
    loadStartTime = time.time()
    trainData, trainLabel, testData = opencsv()
    # print "trainData==>", type(trainData), shape(trainData)
    # print "trainLabel==>", type(trainLabel), shape(trainLabel)
    # print "testData==>", type(testData), shape(testData)
    loadEndTime = time.time()
    print "load data finish"
    print('load data time used:%f' % (loadEndTime - loadStartTime))
    t = time.time()
    # result = knnClassify(trainData, trainLabel, testData)   
    print "finish!"
    k = time.time()
    print('classify time used:%f' % (k - t))


if __name__ == '__main__':
    dRecognition_knn()
