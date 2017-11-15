# **数字识别**

[**数字识别**](/competitions/GettingStarted/DigitRecognizer.md):使用著名的 MNIST 数据来学习计算机视觉基础原理

## 内容说明：

* MNIST（"修改后的国家标准与技术研究所"）是计算机视觉的事实上的 "hello world" 数据集。自1999年发布以来，手写图像的经典数据集已成为基准分类算法的基础。随着新机器学习技术的出现，MNIST 仍然是研究人员和学习者的可靠资源。
* 在本次比赛中，您的目标是正确识别数以万计手写图像的数字。我们策划了一套教程式的内核，涵盖从回归到神经网络的一切。我们鼓励您尝试使用不同的算法来学习第一手什么是有效的，以及技术如何比较。

## 项目规范

> 文档：结尾文件名为项目名.md

* 案例：`competitions/GettingStarted/DigitRecognizer.md`
* 例如：数字识别，文档是属于 `competitions -> GettingStarted` 下面，所以创建 `competitions/GettingStarted` 存放文档就行

> 图片：结尾文件名可自由定义

* 案例：`static/images/comprtitions/GettingStarted/DigitRecognizer/front_page.png`
* 例如：数字识别，图片是属于 `competitions -> GettingStarted -> DigitRecognizer` 下面，所以创建 `competitions/GettingStarted/DigitRecognizer` 存放文档的图片就行


> 代码：结尾文件名可自由定义.py

* 案例：`src/python/GettingStarted/DigitRecognizer/dr_knn_pandas.py`
* 例如：数字识别，代码只有 `竞赛` 有，所以直接创建 `GettingStarted/DigitRecognizer` 存放代码就行
* 要求（方法：完全解耦）
    1. 加载数据
    2. 预处理数据(可没)
    3. 训练模型
    4. 评估模型(可没)
    5. 导出数据

> 数据：结尾文件名可自由定义

* 输入：`datasets/input/GettingStarted/DigitRecognizer/train.csv`
* 输出：`datasets/ouput/GettingStarted/DigitRecognizer/Result_sklearn_knn.csv`
* 例如：数字识别，数据只有 `竞赛` 有，所以直接创建 `GettingStarted/DigitRecognizer` 存放数据就行

## [项目代码](https://github.com/apachecn/kaggle/blob/master/src/python/GettingStarted/DigitRecognizer/dr_knn_pandas.py)

> 1.标注python和编码格式

```python
#!/usr/bin/python
# coding: utf-8
```


> 2.标注项目的描述

```python
'''
Created on 2017-10-26
Update  on 2017-10-26
Author: 【如果是个人】片刻
Team:   【如果是团队】装逼从不退缩（张三、李四 等等）
Github: https://github.com/apachecn/kaggle
'''
```

> 3.示例代码（按照代码的规范来整理就行）

```python
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
    return testLabel


def dRecognition_knn():
    start_time = time.time()

    # 加载数据
    trainData, trainLabel, testData = opencsv()
    print "trainData==>", type(trainData), shape(trainData)
    print "trainLabel==>", type(trainLabel), shape(trainLabel)
    print "testData==>", type(testData), shape(testData)
    print "load data finish"
    stop_time_l = time.time()
    print('load data time used:%f' % (stop_time_l - start_time))

    # 模型训练
    testLabel = knnClassify(trainData, trainLabel, testData)

    # 结果的输出
    saveResult(testLabel, 'datasets/ouput/GettingStarted/DigitRecognizer/Result_sklearn_knn.csv')
    print "finish!"
    stop_time_r = time.time()
    print('classify time used:%f' % (stop_time_r - start_time))


if __name__ == '__main__':
    dRecognition_knn()
```