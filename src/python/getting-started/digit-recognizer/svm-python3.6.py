#!/usr/bin/python3
# coding: utf-8

'''
Created on 2017-10-26
Update  on 2017-10-26
Author: 片刻
Github: https://github.com/apachecn/kaggle
'''

import csv
import time
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


# 加载数据
def opencsv():
    print('Load Data...')
    # 使用 pandas 打开
    dataTrain = pd.read_csv('datasets/getting-started/digit-recognizer/input/train.csv')
    dataPre = pd.read_csv('datasets/getting-started/digit-recognizer/input/test.csv')
    trainData = dataTrain.values[:, 1:]  # 读入全部训练数据
    trainLabel = dataTrain.values[:, 0]
    preData = dataPre.values[:, :]  # 测试全部测试个数据
    return trainData, trainLabel, preData


# 数据预处理-降维
def dRCsv(x_train, x_test, preData, COMPONENT_NUM):
    print('dimensionality reduction...')
    trainData = np.array(x_train)
    testData = np.array(x_test)
    preData = np.array(preData)

    '''
    使用说明：https://www.cnblogs.com/pinard/p/6243025.html
    n_components>=1
      n_components=NUM   设置占特征数量比
    0 < n_components < 1
      n_components=0.99  设置阈值总方差占比
    '''
    pca = PCA(n_components=COMPONENT_NUM, whiten=True)
    pca.fit(trainData)  # Fit the model with X
    pcaTrainData = pca.transform(trainData)  # Fit the model with X and 在X上完成降维.
    pcaTestData = pca.transform(testData)  # Fit the model with X and 在X上完成降维.
    pcaPreData = pca.transform(preData)  # Fit the model with X and 在X上完成降维.

    # pca 方差大小、方差占比、特征数量
    print(pca.explained_variance_, '\n', pca.explained_variance_ratio_, '\n', pca.n_components_)
    print(sum(pca.explained_variance_ratio_))
    return pcaTrainData,  pcaTestData, pcaPreData


# 训练模型
def trainModel(trainData, trainLabel):
    print('Train SVM...')
    svmClf = SVC(C=4, kernel='rbf')
    svmClf.fit(trainData, trainLabel)  # 训练SVM
    return svmClf


# 结果输出保存
def saveResult(result, csvName):
    with open(csvName, 'w') as myFile:
        myWriter = csv.writer(myFile)
        myWriter.writerow(["ImageId", "Label"])
        index = 0
        for r in result:
            index += 1
            myWriter.writerow([index, int(r)])

    print('Saved successfully...')  # 保存预测结果


# 分析数据
def analyse_data(dataMat):
    meanVals = np.mean(dataMat, axis=0)
    meanRemoved = dataMat-meanVals
    covMat = np.cov(meanRemoved, rowvar=0)
    eigvals, eigVects = np.linalg.eig(np.mat(covMat))
    eigValInd = np.argsort(eigvals)

    topNfeat = 100
    eigValInd = eigValInd[:-(topNfeat+1):-1]
    cov_all_score = float(sum(eigvals))
    sum_cov_score = 0
    for i in range(0, len(eigValInd)):
        line_cov_score = float(eigvals[eigValInd[i]])
        sum_cov_score += line_cov_score
        '''
        我们发现其中有超过20%的特征值都是0。
        这就意味着这些特征都是其他特征的副本，也就是说，它们可以通过其他特征来表示，而本身并没有提供额外的信息。

        最前面15个值的数量级大于10^5，实际上那以后的值都变得非常小。
        这就相当于告诉我们只有部分重要特征，重要特征的数目也很快就会下降。

        最后，我们可能会注意到有一些小的负值，他们主要源自数值误差应该四舍五入成0.
        '''
        print('主成分：%s, 方差占比：%s%%, 累积方差占比：%s%%' % (format(i+1, '2.0f'), format(line_cov_score/cov_all_score*100, '4.2f'), format(sum_cov_score/cov_all_score*100, '4.1f')))


# 找出最高准确率
def getOptimalAccuracy(trainData, trainLabel, preData):
    # 分析数据 100个特征左右
    # analyse_data(trainData)
    x_train, x_test, y_train, y_test = train_test_split(trainData, trainLabel, test_size=0.1)
    lineLen, featureLen = np.shape(x_test)
    # print(lineLen, type(lineLen), featureLen, type(featureLen))

    minErr = 1
    minSumErr = 0
    optimalNum = 1
    optimalLabel = []
    optimalSVMClf = None
    pcaPreDataResult = None
    for i in range(30, 45, 1):
        # 评估训练结果
        pcaTrainData,  pcaTestData, pcaPreData = dRCsv(x_train, x_test, preData, i)
        svmClf = trainModel(pcaTrainData, y_train)
        svmtestLabel = svmClf.predict(pcaTestData)

        errArr = np.mat(np.ones((lineLen, 1)))
        sumErrArr = errArr[svmtestLabel != y_test].sum()
        sumErr = sumErrArr/lineLen

        print('i=%s' % i, lineLen, sumErrArr, sumErr)
        if sumErr <= minErr:
            minErr = sumErr
            minSumErr = sumErrArr
            optimalNum = i
            optimalSVMClf = svmClf
            optimalLabel = svmtestLabel
            pcaPreDataResult = pcaPreData
            print("i=%s >>>>> \t" % i, lineLen, int(minSumErr), 1-minErr)

    '''
    展现 准确率与召回率
        precision 准确率
        recall 召回率
        f1-score  准确率和召回率的一个综合得分
        support 参与比较的数量
    参考链接：http://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html#sklearn.metrics.classification_report
    '''

    # target_names 以 y的label分类为准
    # target_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    target_names = [str(i) for i in list(set(y_test))]
    print(target_names)
    print(classification_report(y_test, optimalLabel, target_names=target_names))
    print("特征数量= %s, 存在最优解：>>> \t" % optimalNum, lineLen, int(minSumErr), 1-minErr)
    return optimalSVMClf, pcaPreDataResult


# 存储模型
def storeModel(model, filename):
    import pickle
    with open(filename, 'wb') as fw:
        pickle.dump(model, fw)


# 加载模型
def getModel(filename):
    import pickle
    fr = open(filename, 'rb')
    return pickle.load(fr)


def trainDRSVM():
    startTime = time.time()

    # 加载数据
    trainData, trainLabel, preData = opencsv()
    # 模型训练 (数据预处理-降维)
    optimalSVMClf, pcaPreData = getOptimalAccuracy(trainData, trainLabel, preData)

    storeModel(optimalSVMClf, 'datasets/getting-started/digit-recognizer/ouput/Result_sklearn_SVM.model')
    storeModel(pcaPreData, 'datasets/getting-started/digit-recognizer/ouput/Result_sklearn_SVM.pcaPreData')

    print("finish!")
    stopTime = time.time()
    print('TrainModel store time used:%f s' % (stopTime - startTime))


def preDRSVM():
    startTime = time.time()
    # 加载模型和数据
    optimalSVMClf = getModel('datasets/getting-started/digit-recognizer/ouput/Result_sklearn_SVM.model')
    pcaPreData = getModel('datasets/getting-started/digit-recognizer/ouput/Result_sklearn_SVM.pcaPreData')

    # 结果预测
    testLabel = optimalSVMClf.predict(pcaPreData)
    # print("testLabel = %f" % testscore)
    # 结果的输出
    saveResult(testLabel, 'datasets/getting-started/digit-recognizer/ouput/Result_sklearn_SVM.csv')
    print("finish!")
    stopTime = time.time()
    print('PreModel load time used:%f s' % (stopTime - startTime))


if __name__ == '__main__':
    # 训练并保存模型
    trainDRSVM()

    # 加载预测数据集
    # preDRSVM()
