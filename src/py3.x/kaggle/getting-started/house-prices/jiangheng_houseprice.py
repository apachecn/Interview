# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 17:12:06 2017

@author: Administrator
"""

import pandas as pd
import numpy as np
from pandas import DataFrame, Series
from scipy.stats import boxcox
from sklearn.linear_model import Ridge
import warnings
import os.path

warnings.filterwarnings('ignore')

data_dir = '/opt/data/kaggle/getting-started/house-prices'

# 这里对数据做一些转换,原因要么是某些类别个数太少而且分布相近,要么是特征内的值之间有较为明显的优先级
mapper = {
    'LandSlope': {
        'Gtl': 'Gtl',
        'Mod': 'unGtl',
        'Sev': 'unGtl'
    },
    'LotShape': {
        'Reg': 'Reg',
        'IR1': 'IR1',
        'IR2': 'other',
        'IR3': 'other'
    },
    'RoofMatl': {
        'ClyTile': 'other',
        'CompShg': 'CompShg',
        'Membran': 'other',
        'Metal': 'other',
        'Roll': 'other',
        'Tar&Grv': 'Tar&Grv',
        'WdShake': 'WdShake',
        'WdShngl': 'WdShngl'
    },
    'Heating': {
        'GasA': 'GasA',
        'GasW': 'GasW',
        'Grav': 'Grav',
        'Floor': 'other',
        'OthW': 'other',
        'Wall': 'Wall'
    },
    'HeatingQC': {
        'Po': 1,
        'Fa': 2,
        'TA': 3,
        'Gd': 4,
        'Ex': 5
    },
    'KitchenQual': {
        'Fa': 1,
        'TA': 2,
        'Gd': 3,
        'Ex': 4
    }
}

# 对结果影响很小,或者与其他特征相关性较高的特征将被丢弃
to_drop = [
    'Id', 'Street', 'Utilities', 'Condition2', 'PoolArea', 'PoolQC', 'Fence',
    'YrSold', 'MoSold', 'BsmtHalfBath', 'BsmtFinSF2', 'GarageQual', 'MiscVal',
    'EnclosedPorch', '3SsnPorch', 'GarageArea', 'TotRmsAbvGrd', 'GarageYrBlt',
    'BsmtFinType2', 'BsmtUnfSF', 'GarageCond', 'GarageFinish', 'FireplaceQu',
    'BsmtCond', 'BsmtQual', 'Alley'
]

# 特渣工程之瞎搞特征,别问我思路是什么,纯属乱拍脑袋搞出来,而且对结果貌似也仅有一点点影响
'''
data['house_remod']:  重新装修的年份与房建年份的差值
data['livingRate']:   LotArea查了下是地块面积,这个特征是居住面积/地块面积*总体评价
data['lot_area']:    LotFrontage是与房子相连的街道大小,现在想了下把GrLivArea换成LotArea会不会好点?
data['room_area']:   房间数/居住面积
data['fu_room']:    带有浴室的房间占总房间数的比例
data['gr_room']:    卧室与房间数的占比
'''


def create_feature(data):
    # 是否拥有地下室
    hBsmt_index = data.index[data['TotalBsmtSF'] > 0]
    data['HaveBsmt'] = 0
    data.loc[hBsmt_index, 'HaveBsmt'] = 1
    data['house_remod'] = data['YearRemodAdd'] - data['YearBuilt']
    data['livingRate'] = (data['GrLivArea'] /
                          data['LotArea']) * data['OverallCond']
    data['lot_area'] = data['LotFrontage'] / data['GrLivArea']
    data['room_area'] = data['TotRmsAbvGrd'] / data['GrLivArea']
    data['fu_room'] = data['FullBath'] / data['TotRmsAbvGrd']
    data['gr_room'] = data['BedroomAbvGr'] / data['TotRmsAbvGrd']


def processing(data):
    # 构造新特征
    create_feature(data)
    # 丢弃特征
    data.drop(to_drop, axis=1, inplace=True)

    # 填充None值,因为在特征说明中,None也是某些特征的一个值,所以对于这部分特征的缺失值以None填充
    fill_none = ['MasVnrType', 'BsmtExposure', 'GarageType', 'MiscFeature']
    for col in fill_none:
        data[col].fillna('None', inplace=True)

    # 对其他缺失值进行填充,离散型特征填充众数,数值型特征填充中位数
    na_col = data.dtypes[data.isnull().any()]
    for col in na_col.index:
        if na_col[col] != 'object':
            med = data[col].median()
            data[col].fillna(med, inplace=True)
        else:
            mode = data[col].mode()[0]
            data[col].fillna(mode, inplace=True)

    # 对正态偏移的特征进行正态转换,numeric_col就是数值型特征,zero_col是含有零值的数值型特征
    # 因为如果对含零特征进行转换的话会有各种各种的小问题,所以干脆单独只对非零数值进行转换
    numeric_col = data.skew().index
    zero_col = data.columns[data.isin([0]).any()]
    for col in numeric_col:
        # 对于那些condition特征,例如取值是0,1,2,3...那些我不作变换,因为意义不大
        if len(pd.value_counts(data[col])) <= 10: continue
        # 如果是含有零值的特征,则只对非零值变换,至于用哪种形式变换,boxcox会自动根据数据来调整
        if col in zero_col:
            trans_data = data[data > 0][col]
            before = abs(trans_data.skew())
            cox, _ = boxcox(trans_data)
            log_after = abs(Series(cox).skew())
            if log_after < before:
                data.loc[trans_data.index, col] = cox
        # 如果是非零值的特征,则全部作转换
        else:
            before = abs(data[col].skew())
            cox, _ = boxcox(data[col])
            log_after = abs(Series(cox).skew())
            if log_after < before:
                data.loc[:, col] = cox
    # mapper值的映射转换
    for col, mapp in mapper.items():
        data.loc[:, col] = data[col].map(mapp)


df_train = pd.read_csv(os.path.join(data_dir, "train.csv"))
df_test = pd.read_csv(os.path.join(data_dir, "test.csv"))
test_ID = df_test['Id']

# 去除离群点
GrLivArea_outlier = set(df_train.index[(df_train['SalePrice'] < 200000) & (
    df_train['GrLivArea'] > 4000)])
LotFrontage_outlier = set(df_train.index[df_train['LotFrontage'] > 300])
df_train.drop(LotFrontage_outlier | GrLivArea_outlier, inplace=True)

# 因为删除了几行数据,所以index的序列不再连续,需要重新reindex
df_train.reset_index(drop=True, inplace=True)
prices = np.log1p(df_train.loc[:, 'SalePrice'])
df_train.drop(['SalePrice'], axis=1, inplace=True)
# 这里对训练集和测试集进行合并,然后再进行特征工程
all_data = pd.concat([df_train, df_test])
all_data.reset_index(drop=True, inplace=True)

# 进行特征工程
processing(all_data)

# dummy转换
dummy = pd.get_dummies(all_data, drop_first=True)

# 试了Ridge,Lasso,ElasticNet以及GBM,发现ridge的表现比其他的都好,参数alpha=6是调参结果
ridge = Ridge(6)
ridge.fit(dummy.iloc[:prices.shape[0], :], prices)
result = np.expm1(ridge.predict(dummy.iloc[prices.shape[0]:, :]))
pre = DataFrame(result, columns=['SalePrice'])
prediction = pd.concat([test_ID, pre], axis=1)
prediction.to_csv(os.path.join(data_dir, "submission_1.csv"), index=False)
