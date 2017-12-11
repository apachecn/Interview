## **房价预测**

***

[房价预测](https://www.kaggle.com/c/house-prices-advanced-regression-techniques):Predict sales prices and practice feature engineering

### **内容说明**

- Ask a home buyer to describe their dream house, and they probably won't begin with the height of the basement ceiling or the proximity to an east-west railroad. But this playground competition's dataset proves that much more influences price negotiations than the number of bedrooms or a white-picket fence.

- With 79 explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa, this competition challenges you to predict the final price of each home.

### **开发流程**

>收集数据:[数据集](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)


>准备数据:
```python
def dataProcess(df_train, df_test):
    trainLabel = df_train['SalePrice']
    df = pd.concat((df_train,df_test), axis=0, ignore_index=True)
    df.dropna(axis=1, inplace=True)
    df = pd.get_dummies(df)
    trainData = df[:df_train.shape[0]]
    test = df[df_train.shape[0]:]
    return trainData, trainLabel, test 
```

>模型训练：产生训练模型:
```
暂时没写
```

>模型评估:RMSE
```
暂时没写
```

>结果预测：
```python
def ridgeRegression(trainData, trainLabel, df_test):
    ridge = Ridge(alpha=10.0)   # default:k = 5,defined by yourself:KNeighborsClassifier(n_neighbors=10)
    ridge.fit(trainData, trainLabel)
    predict = ridge.predict(df_test)
    pred_df = pd.DataFrame(predict, index=df_test["Id"], columns=["SalePrice"])
    return pred_df 
```

>结果导出：
```python
def saveResult(result):
    result.to_csv(os.path.join(data_dir,"submission.csv" ), sep=',', encoding='utf-8')
```