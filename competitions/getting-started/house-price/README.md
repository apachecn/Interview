# **房价预测**

![](/static/images/competitions/getting-started/house-price/housesbanner.png)

## 比赛说明

* [**房价预测**](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) 
* 要求购房者描述他们的梦想之家，他们可能不会从地下室天花板的高度或与东西方铁路的接近度开始。但是这个游乐场比赛的数据集证明，对价格谈判的影响远远超过卧室或白色栅栏的数量。
* 有79个解释变量描述（几乎）爱荷华州埃姆斯的住宅房屋的每个方面，这个竞赛挑战你预测每个房屋的最终价格。

## 参赛成员

* 开源组织: [ApacheCN ~ apachecn.org](http://www.apachecn.org/)

## 比赛分析

* 回归问题：价格的问题
* 常用算法： `回归`、`树回归`、`GBDT`、`xgboost`、`lightGBM`

```
步骤:
一. 数据分析
1. 下载并加载数据
2. 总体预览:了解每列数据的含义,数据的格式等
3. 数据初步分析,使用统计学与绘图:初步了解数据之间的相关性,为构造特征工程以及模型建立做准备

二. 特征工程
1.根据业务,常识,以及第二步的数据分析构造特征工程.
2.将特征转换为模型可以辨别的类型(如处理缺失值,处理文本进行等)

三. 模型选择
1.根据目标函数确定学习类型,是无监督学习还是监督学习,是分类问题还是回归问题等.
2.比较各个模型的分数,然后取效果较好的模型作为基础模型.

四. 模型融合

五. 修改特征和模型参数
1.可以通过添加或者修改特征,提高模型的上限.
2.通过修改模型的参数,是模型逼近上限
```

## 一. 数据分析

### 数据下载和加载

* 数据集下载地址：<https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data>

```python
# 导入相关数据包
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
```

### 特征说明

