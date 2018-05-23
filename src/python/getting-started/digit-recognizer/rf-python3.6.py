#!/usr/bin/python
# coding: utf-8
'''
Created on 2018-05-14
Update  on 2018-05-19
Author: 平淡的天/wang-sw
Github: https://github.com/apachecn/kaggle
'''
import os.path
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
data_dir = 'G:/data/kaggle/datasets/getting-started/digit-recognizer/'

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
      n_components=NUM   设置降维到的维度数目
    0 < n_components < 1
      n_components=0.99  设置阈值(总方差占比)决定降维到的维度数目
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
def trainModel(X_train, y_train):
    print('Train RF...')
    clf = RandomForestClassifier(
        n_estimators=10,
        max_depth=10,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=34)
    clf.fit(X_train, y_train)  # 训练rf

    # clf=LGBMClassifier(num_leaves=63, max_depth=7, n_estimators=80, n_jobs=20)

    # param_test1 = {'n_estimators':arange(10,150,10),'max_depth':arange(1,21,1)}
    # gsearch1 = GridSearchCV(estimator = clf, param_grid = param_test1, scoring='accuracy',iid=False,cv=5)
    # gsearch1.fit(X_train, y_train)
    # print(gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_)
    # clf=gsearch1.best_estimator_

    return clf


# 计算准确率
def printAccuracy(y_test ,y_predict):
    zeroLable = y_test - y_predict
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
    n = len(result)
    print('the size of test set is {}'.format(n))
    with open(os.path.join(data_dir, 'output/Result_sklearn_RF.csv'), 'w') as fw:
        fw.write('{},{}\n'.format('ImageId', 'Label'))
        for i in range(1, n + 1):
            fw.write('{},{}\n'.format(i, result[i - 1]))
    print('Result saved successfully... and the path = {}'.format(csvName))


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

    X_train, X_test, y_train, y_test = train_test_split(
        data_pca[0:len(train_data)], label, test_size=0.1, random_state=34)

    rfClf = trainModel(X_train, y_train)

    # 保存结果
    storeModel(data_pca[len(train_data):], os.path.join(data_dir, 'output/Result_sklearn_rf.pcaPreData'))
    storeModel(rfClf, os.path.join(data_dir, 'output/Result_sklearn_rf.model'))

    # 模型准确率
    y_predict = rfClf.predict(X_test)
    printAccuracy(y_test, y_predict)

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
    saveResult(result, os.path.join(data_dir, 'output/Result_sklearn_rf.csv'))
    print("finish!")
    stopTime = time.time()
    print('PreModel load time used:%f s' % (stopTime - startTime))


if __name__ == '__main__':

    # 训练并保存模型
    trainRF()

    # 加载预测数据集
    preRF()
