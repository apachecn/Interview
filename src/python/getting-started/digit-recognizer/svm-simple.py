#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import time
import pandas as pd
import numpy as np
from numpy import *
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split


# 加载数据
def opencsv():
    print('Load Data...')
    # 使用 pandas 打开
    dataTrain = pd.read_csv(r'../../../../datasets/getting-started/digit-recognizer/input/train.csv')
    dataTest = pd.read_csv(r'../../../../datasets/getting-started/digit-recognizer/input/test.csv')

    trainData = dataTrain.values[:, 1:]  # 读入全部训练数据
    trainLabel = dataTrain.values[:, 0]
    preData = dataTest.values[:, :]  # 测试全部测试个数据
    return trainData, trainLabel,preData


def dRCsv(x_train, x_test, preData, COMPONENT_NUM):
    print('dimensionality reduction...')
    trainData = np.array(x_train)
    testData = np.array(x_test)
    preData = np.array(preData)
    pca = PCA(n_components=COMPONENT_NUM, whiten=True)
    pca.fit(trainData)  # Fit the model with X
    pcaTrainData = pca.transform(trainData)  # Fit the model with X and 在X上完成降维.
    pcaTestData = pca.transform(testData)  # Fit the model with X and 在X上完成降维.
    pcaPreData = pca.transform(preData)  # Fit the model with X and 在X上完成降维.
    print(sum(pca.explained_variance_ratio_))
    return pcaTrainData,  pcaTestData, pcaPreData


def svmClassify(trainData, trainLabel):
     print('Train SVM...')
     svmClf=SVC(C=4, kernel='rbf')
     svmClf.fit(trainData, trainLabel)  # 训练SVM
     return svmClf


def saveResult(result, csvName):
     with open(csvName, 'w',newline='') as myFile:
         myWriter = csv.writer(myFile)
         myWriter.writerow(["ImageId", "Label"])
         index = 0
         for i in result:
            # tmp = []
            index = index+1
            # tmp.append(index)
            # tmp.append(i)
            # tmp.append(int(i))
            myWriter.writerow([index, int(i)])


def SVM():
     #加载数据
     start_time = time.time()
     trainData, trainLable,preData=opencsv()
     print("load data finish")
     stop_time_l = time.time()
     print('load data time used:%f' % (stop_time_l - start_time))
     trainData, testData,trainLable,testLabletrue = train_test_split(trainData, trainLable, test_size=0.1, random_state=41)#交叉验证 测试集10%
    
     #pca降维
     trainData,testData,preData =dRCsv(trainData,testData,preData,35)  
     # print (trainData,trainLable)


     # 模型训练
     svmClf=svmClassify(trainData, trainLable)
     print ('trainsvm finished')


     # 结果预测
     testLable=svmClf.predict(testData)
     preLable=svmClf.predict(preData)
    
    


     #交叉验证
     zeroLable=testLabletrue-testLable
     rightCount=0
     for i in range(len(zeroLable)):
       if zeroLable[i]==0:
          rightCount+=1
     print ('the right rate is:',float(rightCount)/len(zeroLable))


     # 结果的输出
     saveResult(preLable, r'../../../../datasets/getting-started/digit-recognizer/ouput/Result_sklearn_SVM.csv')
     print( "finish!")
     stop_time_r = time.time()
     print('classify time used:%f' % (stop_time_r - start_time))
    

if __name__ == '__main__':
     SVM()

  