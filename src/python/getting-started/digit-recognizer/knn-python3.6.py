#!/usr/bin/python
# coding: utf-8
'''
Created on 2017-10-26
Update  on 2018-05-16
Author: 片刻/ccyf00
Github: https://github.com/apachecn/kaggle
'''
import os
import csv
import datetime
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier


data_dir = '/opt/data/kaggle/getting-started/digit-recognizer/'


# 加载数据
def opencsv():
    # 使用 pandas 打开
    data = pd.read_csv(os.path.join(data_dir, 'input/train.csv'))
    data1 = pd.read_csv(os.path.join(data_dir, 'input/test.csv'))

    train_data = data.values[:, 1:]  # 读入全部训练数据,  [行，列]
    train_label = data.values[:, 0]  # 读取列表的第一列
    test_data = data1.values[:, :]  # 测试全部测试个数据
    return train_data, train_label, test_data


# 数据预处理-降维 PCA主成成分分析
def dRPCA(x_train, x_test, COMPONENT_NUM):
    print('dimensionality reduction...')
    trainData = np.array(x_train)
    testData = np.array(x_test)
    '''
    使用说明：https://www.cnblogs.com/pinard/p/6243025.html
    n_components>=1
      n_components=NUM   设置占特征数量比
    0 < n_components < 1
      n_components=0.99  设置阈值总方差占比
    '''
    pca = PCA(n_components=COMPONENT_NUM, whiten=False)
    pca.fit(trainData)  # Fit the model with X
    pcaTrainData = pca.transform(trainData)  # Fit the model with X and 在X上完成降维.
    pcaTestData = pca.transform(testData)  # Fit the model with X and 在X上完成降维.

    # pca 方差大小、方差占比、特征数量
    # print("方差大小:\n", pca.explained_variance_, "方差占比:\n", pca.explained_variance_ratio_)
    print("特征数量: %s" % pca.n_components_)
    print("总方差占比: %s" % sum(pca.explained_variance_ratio_))
    return pcaTrainData, pcaTestData


def trainModel(trainData, trainLabel):
    clf = KNeighborsClassifier()  # default:k = 5,defined by yourself:KNeighborsClassifier(n_neighbors=10)
    clf.fit(trainData, np.ravel(trainLabel))  # ravel Return a contiguous flattened array.
    return clf


def saveResult(result, csvName):
    with open(csvName, 'w') as myFile:  # 创建记录输出结果的文件（w 和 wb 使用的时候有问题）
        # python3里面对 str和bytes类型做了严格的区分，不像python2里面某些函数里可以混用。所以用python3来写wirterow时，打开文件不要用wb模式，只需要使用w模式，然后带上newline=''
        myWriter = csv.writer(myFile)
        myWriter.writerow(["ImageId", "Label"])
        index = 0
        for r in result:
            index += 1
            myWriter.writerow([index, int(r)])
    print('Saved successfully...')  # 保存预测结果


def dRecognition_knn():
    # 开始时间
    sta_time = datetime.datetime.now()

    # 加载数据
    trainData, trainLabel, testData = opencsv()
    # print("trainData==>", type(trainData), shape(trainData))
    # print("trainLabel==>", type(trainLabel), shape(trainLabel))
    # print("testData==>", type(testData), shape(testData))
    print("load data finish")
    end_time_1 = datetime.datetime.now()
    print('load data time used: %s' % end_time_1)

    # 降维处理
    trainDataPCA, testDataPCA = dRPCA(trainData, testData, 0.8)

    # 模型训练
    clf = trainModel(trainDataPCA, trainLabel)
    # 结果预测
    testLabel = clf.predict(testDataPCA)

    # 结果的输出
    saveResult(testLabel, os.path.join(data_dir, 'output/Result_knn.csv'))
    print("finish!")

    # 结束时间
    end_time = datetime.datetime.now()
    times = (end_time - sta_time).seconds
    print("\n运行时间: %ss == %sm == %sh\n\n" % (times, times/60, times/60/60))


if __name__ == '__main__':
    dRecognition_knn()
