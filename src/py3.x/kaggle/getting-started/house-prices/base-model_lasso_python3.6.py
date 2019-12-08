#!/usr/bin/python
# coding: utf-8
'''
Created on 2017-12-11
Update  on 2017-12-11
Author: Usernametwo
Github: https://github.com/apachecn/kaggle
'''
import time
import pandas as pd
from sklearn.linear_model import Ridge
import os.path

data_dir = '/opt/data/kaggle/getting-started/house-prices'


# 加载数据
def opencsv():
    # 使用 pandas 打开
    df_train = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    df_test = pd.read_csv(os.path.join(data_dir, 'test.csv'))

    return df_train, df_test


def saveResult(result):
    result.to_csv(
        os.path.join(data_dir, "submission.csv"), sep=',', encoding='utf-8')


def ridgeRegression(trainData, trainLabel, df_test):
    ridge = Ridge(
        alpha=10.0
    )  # default:k = 5,defined by yourself:KNeighborsClassifier(n_neighbors=10)
    ridge.fit(trainData, trainLabel)
    predict = ridge.predict(df_test)
    pred_df = pd.DataFrame(predict, index=df_test["Id"], columns=["SalePrice"])
    return pred_df


def dataProcess(df_train, df_test):
    trainLabel = df_train['SalePrice']
    df = pd.concat((df_train, df_test), axis=0, ignore_index=True)
    df.dropna(axis=1, inplace=True)
    df = pd.get_dummies(df)
    trainData = df[:df_train.shape[0]]
    test = df[df_train.shape[0]:]
    return trainData, trainLabel, test


def Regression_ridge():
    start_time = time.time()

    # 加载数据
    df_train, df_test = opencsv()

    print("load data finish")
    stop_time_l = time.time()
    print('load data time used:%f' % (stop_time_l - start_time))

    # 数据预处理
    train_data, trainLabel, df_test = dataProcess(df_train, df_test)

    # 模型训练预测
    result = ridgeRegression(train_data, trainLabel, df_test)

    # 结果的输出
    saveResult(result)
    print("finish!")
    stop_time_r = time.time()
    print('classify time used:%f' % (stop_time_r - start_time))


if __name__ == '__main__':
    Regression_ridge()
