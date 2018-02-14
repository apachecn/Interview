#!/usr/bin/python
# coding: utf-8
'''
Created on 2017-10-26
Update  on 2017-10-26
Author: 片刻
Github: https://github.com/apachecn/kaggle
'''

# 导入相关数据包
import numpy as np
import pandas as pd
# import seaborn as sns
import matplotlib.pyplot as plt

root_path = '/opt/git/kaggle/datasets/getting-started/titanic/input'

train_data = pd.read_csv('%s/%s' % (root_path, 'train.csv'))
test_data = pd.read_csv('%s/%s' % (root_path, 'test.csv'))


print(train_data.head(5))
print(train_data.info())
# # 返回数值型变量的统计量
print(train_data.describe())
