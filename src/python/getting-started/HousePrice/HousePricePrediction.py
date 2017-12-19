#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn import linear_model
import csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso,LinearRegression,Ridge,ElasticNet,TheilSenRegressor,HuberRegressor,RANSACRegressor,LassoCV
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
import itertools
from sklearn.tree import DecisionTreeRegressor,ExtraTreeRegressor
from sklearn.ensemble import AdaBoostRegressor,ExtraTreesRegressor,GradientBoostingRegressor,RandomForestRegressor
from xgboost import XGBRegressor
from scipy.stats import skew
	
def calScore(pre,label):
	if len(pre)==len(label):
		score=np.sqrt(sum(pow(x-y,2) for x,y in zip(pre,label))/len(label))
		print score
	else:
		print "Their length are different"
def loadData():
	train=pd.read_csv('train.csv')
	test=pd.read_csv('test.csv')
	#id=train['Id']
	#train['Id']
	#print train.head(3)
	#由于将全部的数值型变量映射为正太分布的
	train["SalePrice"] = np.log1p(train["SalePrice"])
	label=train['SalePrice']
	del train['SalePrice']
	
	#将训练集与测试集融合
	train=pd.concat([train,test])
	#索引为列的名字，值为类型
	#数值型特征
	numeric_feats = train.dtypes[train.dtypes != "object"].index
	skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) #计算偏度
	skewed_feats = skewed_feats[skewed_feats > 0.75]	#偏度大于0.75的进行正态变换
	skewed_feats = skewed_feats.index
	train[skewed_feats] = np.log1p(train[skewed_feats])
	#分类特征进行哑编码
	train=pd.get_dummies(train)
	#用均值填充空值
	train=train.fillna(train.mean())
	test=train[train['Id']>=1461]
	train=train[train['Id']<1461]
	del train['Id']
	sub=test[['Id']]
	del test['Id']	
	
	
	#模型选择
	X_train,X_test,Y_train,Y_test=train_test_split(train,label,test_size=0.33)
	
	regs = [
    ['LassoCV',LassoCV(alphas = [1, 0.1, 0.001, 0.0005])],
    ['LinearRegression',LinearRegression()],
    ['Ridge',Ridge()],
    ['ElasticNet',ElasticNet()],
    ['RANSACRegressor',RANSACRegressor()],
    ['HuberRegressor',HuberRegressor()],
    ['DecisionTreeRegressor',DecisionTreeRegressor()],
    ['ExtraTreeRegressor',ExtraTreeRegressor()],
    ['AdaBoostRegressor',AdaBoostRegressor(n_estimators=150)],
    ['ExtraTreesRegressor',ExtraTreesRegressor(n_estimators=150)],
    ['GradientBoostingRegressor',GradientBoostingRegressor(n_estimators=150)],
    ['RandomForestRegressor',RandomForestRegressor(n_estimators=150)],
    ['XGBRegressor',XGBRegressor(n_estimators=360, max_depth=2, learning_rate=0.1)],
	]
	
	preds=[]
	for reg_name,reg in regs:
		print reg_name
		reg.fit(X_train,Y_train)
		y_pred=reg.predict(X_test)
		if np.sum(y_pred<0)>0:
			print 'y_pred have '+str(np.sum(y_pred<0))+" are negtive, we replace it witt median value of y_pred"
			y_pred[y_pred<0]=np.median(y_pred)
		score=np.sqrt(mean_squared_error(np.log(y_pred),np.log(Y_test)))
		print 
		preds.append([reg_name,y_pred])
		
	final_results=[]
	for comb_len in range(1,len(regs)+1):
		print "Model num:"+str(comb_len)
		results=[]
		for comb in itertools.combinations(preds,comb_len):
			#选取一个模型的组合，比如comb_len=2的时候，comb为(['Lasso',y_pred],['Ridge',y_pred]
			pred_sum=0
			model_name=[]
			for reg_name,pre in comb:
				pred_sum+=pre
				model_name.append(reg_name)
			pred_sum/=comb_len
			model_name='+'.join(model_name)
			score=np.sqrt(mean_squared_error(np.log(np.expm1(pred_sum)),np.log(np.expm1(Y_test))))
			results.append([model_name,score])
		#操作每一个融合模型的分数
		results=sorted(results,key=lambda x:x[1])
		for model_name,score in results:
			print model_name+":"+str(score)
		print 
		final_results.append(results[0])
		
		
	print "best set of models"
	print 
	for i in final_results:
		print i
	
	
	#选择模型
	result=0
	choose_model=[LassoCV(alphas = [1, 0.1, 0.001, 0.0005]),GradientBoostingRegressor(n_estimators=150),XGBRegressor(n_estimators=360, max_depth=2, learning_rate=0.1)]
	for model in choose_model:
		reg=model.fit(train,label)
		pre=reg.predict(test)
		result+=pre
	result/=3

	#写入文件
	result=np.expm1(result)
	sub['SalePrice']=result
	list=[[int(x[0]),x[1]] for x in sub.values]
	with open("submission.csv",'wb') as f:
		writer=csv.writer(f)
		writer.writerow(['Id','SalePrice'])
		for i in range(len(list)):
			writer.writerow(list[i])

def main():
	loadData()

if __name__=='__main__':
	main()
