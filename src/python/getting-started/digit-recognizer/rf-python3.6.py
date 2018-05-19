#!/usr/bin/python
# coding: utf-8
'''
Created on 2018-05-14
Update  on 2018-05-19
Author: 平淡的天/wang-sw
Github: https://github.com/apachecn/kaggle
'''

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
# from sklearn.model_selection import GridSearchCV
# from numpy import arange
# from lightgbm import LGBMClassifier
import os.path
import time

# 数据路径
data_dir = '/media/wsw/B634091A3408DF6D/data/kaggle/datasets/getting-started/digit-recognizer/'

# 加载数据
def opencsv():
    # 使用 pandas 打开
    train_data = pd.read_csv(os.path.join(data_dir, 'input/train.csv'))
    test_data = pd.read_csv(os.path.join(data_dir, 'input/test.csv'))
    data = pd.concat([train_data, test_data], axis=0).reset_index(drop=True)
    data.drop(['label'], axis=1, inplace=True)
    label = train_data.label
    return train_data,test_data,data, label

# 数据预处理-降维 PCA主成成分分析
def dRPCA(data, COMPONENT_NUM=100):
    print('dimensionality reduction...')
    data = np.array(data)
    '''
    使用说明：https://www.cnblogs.com/pinard/p/6243025.html
    n_components>=1
      n_components=NUM   设置占特征数量
    0 < n_components < 1
      n_components=0.99  设置阈值总方差占比
    '''
    pca = PCA(n_components=COMPONENT_NUM, random_state=34)
    data_pca = pca.fit_transform(data)

    # pca 方差大小、方差占比、特征数量
    print(pca.explained_variance_, '\n', pca.explained_variance_ratio_, '\n',
          pca.n_components_)
    print(sum(pca.explained_variance_ratio_))
    storeModel(data_pca, os.path.join(data_dir, 'output/Result_sklearn_rf.pcaData'))
    return data_pca


# 训练模型
def trainModel(Xtrain, xtest):
    print('Train RF...')
    clf = RandomForestClassifier(
        n_estimators=140,
        max_depth=20,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=34)
    clf.fit(Xtrain, xtest)  # 训练rf

    # clf=LGBMClassifier(num_leaves=63, max_depth=7, n_estimators=80, n_jobs=20)

    # param_test1 = {'n_estimators':arange(10,150,10),'max_depth':arange(1,21,1)}
    # gsearch1 = GridSearchCV(estimator = clf, param_grid = param_test1, scoring='accuracy',iid=False,cv=5)
    # gsearch1.fit(Xtrain,xtest)
    # print(gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_)
    # clf=gsearch1.best_estimator_

    return clf


# 计算准确率
def printAccuracy(ytest ,y_predict):
    zeroLable = ytest - y_predict
    rightCount = 0
    for i in range(len(zeroLable)):
        if list(zeroLable)[i] == 0:
            rightCount += 1
    print('the right rate is:', float(rightCount) / len(zeroLable))

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

# 结果输出保存
def saveResult(result, csvName):
    i = 0
    fw = open(csvName, 'w')
    with open(os.path.join(data_dir, 'output/sample_submission.csv')
              ) as pred_file:
        fw.write('{},{}\n'.format('ImageId', 'Label'))
        for line in pred_file.readlines()[1:]:
            splits = line.strip().split(',')
            fw.write('{},{}\n'.format(splits[0], result[i]))
            i += 1
    fw.close()
    print('Result saved successfully...')


def trainRF():
    start_time = time.time()
    # 加载数据
    train_data, test_data, data, label = opencsv()
    print("load data finish")
    stop_time_l = time.time()
    print('load data time used:%f s' % (stop_time_l - start_time))

    startTime = time.time()
    # 模型训练 (数据预处理-降维)
    data_pca = dRPCA(data,100)

    Xtrain, Ytrain, xtest, ytest = train_test_split(
        data_pca[0:len(train_data)], label, test_size=0.1, random_state=34)

    rfClf = trainModel(Xtrain, xtest)

    # 保存结果
    storeModel(data_pca[len(train_data):], os.path.join(data_dir, 'output/Result_sklearn_rf.pcaPreData'))
    storeModel(rfClf, os.path.join(data_dir, 'output/Result_sklearn_rf.model'))

    # 模型准确率
    y_predict = rfClf.predict(Ytrain)
    printAccuracy(ytest, y_predict)

    print("finish!")
    stopTime = time.time()
    print('TrainModel store time used:%f s' % (stopTime - startTime))


def preRF():
    startTime = time.time()
    # 加载模型和数据
    clf=getModel(os.path.join(data_dir, 'output/Result_sklearn_rf.model'))
    pcaPreData = getModel(os.path.join(data_dir, 'output/Result_sklearn_rf.pcaPreData'))

    # 结果预测
    result = clf.predict(pcaPreData)

    # 结果的输出
    saveResult(result,os.path.join(data_dir, 'output/Result_sklearn_rf.csv'))
    print("finish!")
    stopTime = time.time()
    print('PreModel load time used:%f s' % (stopTime - startTime))


if __name__ == '__main__':

    # 训练并保存模型
    trainRF()

    # 加载预测数据集
    preRF()