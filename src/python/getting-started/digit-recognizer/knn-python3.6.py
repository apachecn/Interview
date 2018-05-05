#!/usr/bin/python
# coding: utf-8
'''
Created on 2017-10-26
Update  on 2017-10-26
Author: 片刻
Github: https://github.com/apachecn/kaggle
'''

import csv
import time
import pandas as pd
from numpy import shape, ravel
from sklearn.neighbors import KNeighborsClassifier


# 加载数据
def opencsv():
    # 使用 pandas 打开
    data = pd.read_csv(
        'datasets/getting-started/digit-recognizer/input/train.csv')
    data1 = pd.read_csv(
        'datasets/getting-started/digit-recognizer/input/test.csv')

    train_data = data.values[0:, 1:]  # 读入全部训练数据,  [行，列]
    train_label = data.values[0:, 0] # 读取列表的第一列
    test_data = data1.values[0:, 0:]  # 测试全部测试个数据
    return train_data, train_label, test_data


def saveResult(result, csvName):
    with open(csvName, 'w',newline='') as myFile: # 创建记录输出结果的文件（w 和 wb 使用的时候有问题）
    #python3里面对 str和bytes类型做了严格的区分，不像python2里面某些函数里可以混用。所以用python3来写wirterow时，打开文件不要用wb模式，只需要使用w模式，然后带上newline=''
        myWriter = csv.writer(myFile) # 对文件执行写入
        myWriter.writerow(["ImageId", "Label"]) # 设置表格的列名
        index = 0
        for i in result:
            tmp = []
            index = index + 1
            tmp.append(index)
            # tmp.append(i)
            tmp.append(int(i)) # 测试集的标签值
            myWriter.writerow(tmp)


def knnClassify(trainData, trainLabel):
    knnClf = KNeighborsClassifier()   # default:k = 5,defined by yourself:KNeighborsClassifier(n_neighbors=10)
    knnClf.fit(trainData, ravel(trainLabel)) # ravel Return a contiguous flattened array.
    return knnClf


def dRecognition_knn():
    start_time = time.time()

    # 加载数据
    trainData, trainLabel, testData = opencsv()
    # print("trainData==>", type(trainData), shape(trainData))
    # print("trainLabel==>", type(trainLabel), shape(trainLabel))
    # print("testData==>", type(testData), shape(testData))
    print("load data finish")
    stop_time_l = time.time()
    print('load data time used:%f' % (stop_time_l - start_time))

    # 模型训练
    knnClf = knnClassify(trainData, trainLabel)

    # 结果预测
    testLabel = knnClf.predict(testData)

    # 结果的输出
    saveResult(
        testLabel,
        'datasets/getting-started/digit-recognizer/ouput/Result_sklearn_knn.csv'
    )
    print("finish!")
    stop_time_r = time.time()
    print('classify time used:%f' % (stop_time_r - start_time))


if __name__ == '__main__':
    dRecognition_knn()
