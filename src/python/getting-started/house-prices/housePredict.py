#!/usr/bin/env python3
#-*- coding:utf-8 -*-
'''
Created on 2017-12-2
Update  on 2017-12-2
Author: loveSnowBest
Github: https://github.com/zehuichen123/kaggle
'''

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

rawData=pd.read_csv('train.csv')
testData=pd.read_csv('test.csv')
testId=testData['Id']
X_test=testData.drop(['Id'],axis=1)

Y_train=rawData['SalePrice']
X_train=rawData.drop(['SalePrice','Id'],axis=1)

X=pd.concat([X_train,X_test],axis=0,keys={'train','test'},ignore_index=False)

X_d=pd.get_dummies(X)

keep_cols=X_d.select_dtypes(include=['number']).columns
X_d=X_d[keep_cols]

X_train=X_d.loc['train']
X_test=X_d.loc['test']

X_train=X_train.fillna(X_train.mean())
X_test=X_test.fillna(X_test.mean())

ss=StandardScaler()
X_scale=ss.fit_transform(X_train)
X_test_scale=ss.transform(X_test)

rr=GradientBoostingRegressor(n_estimators=3000,learning_rate=0.05, max_features='sqrt')

rr.fit(X_scale,Y_train)
predict=np.array(rr.predict(X_test_scale))
final=np.hstack((testId.reshape(-1,1),predict.reshape(-1,1)))
np.savetxt('new.csv',final,delimiter=',',fmt='%d')
