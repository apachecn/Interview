
# House Prices: Advanced Regression Techniques in Kaggle 

*author: loveSnowBest*  

## 1. A brief introduction to this competition
This competition is a getting started one. As the title shows us, what we need to use for this competition is regression model. Here is the official description about this compeition:
> Ask a home buyer to describe their dream house, and they probably won't begin with the height of the basement ceiling or the proximity to an east-west railroad. But this playground competition's dataset proves that much more influences price negotiations than the number of bedrooms or a white-picket fence.
With 79 explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa, this competition challenges you to predict the final price of each home.

## 2. My solution

### import what we need



```python
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
```

### load the data


```python
rawData=pd.read_csv('train.csv')
testData=pd.read_csv('test.csv')
```

And let's have a look at our data use head method:


```Python
rawData.head()
```



<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>MSSubClass</th>
      <th>MSZoning</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>Street</th>
      <th>Alley</th>
      <th>LotShape</th>
      <th>LandContour</th>
      <th>Utilities</th>
      <th>...</th>
      <th>PoolArea</th>
      <th>PoolQC</th>
      <th>Fence</th>
      <th>MiscFeature</th>
      <th>MiscVal</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>SaleType</th>
      <th>SaleCondition</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>60</td>
      <td>RL</td>
      <td>65.0</td>
      <td>8450</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>2</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>208500</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>20</td>
      <td>RL</td>
      <td>80.0</td>
      <td>9600</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>5</td>
      <td>2007</td>
      <td>WD</td>
      <td>Normal</td>
      <td>181500</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>60</td>
      <td>RL</td>
      <td>68.0</td>
      <td>11250</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>9</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>223500</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>70</td>
      <td>RL</td>
      <td>60.0</td>
      <td>9550</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>2</td>
      <td>2006</td>
      <td>WD</td>
      <td>Abnorml</td>
      <td>140000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>60</td>
      <td>RL</td>
      <td>84.0</td>
      <td>14260</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>12</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>250000</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 81 columns</p>
</div>



### split original data into X,Y
First, we use drop method to split rawData into X and Y. Since we need to give Id for prediction, we should save testId before we drop them and finally we put it back.


```python
Y_train=rawData['SalePrice']
X_train=rawData.drop(['SalePrice','Id'],axis=1)

testId=testData['Id']
X_test=testData.drop(['Id'],axis=1)
```

### deal with categorical data
In scikit, we can use DictVectorizer and in pandas we can just use get_dummies. Here I choose the latter one. To use dummies we should put the X_train and X_test together.


```python
# add new keys train and test for the convienence of the future split
X=pd.concat([X_train,X_test],axis=0,keys={'first','second'},
            ignore_index=False)
X_d=pd.get_dummies(X)
```

DO NOT forget to drop the original categorical data for pandas won't help you drop them automatically. You need to drop it manually:


```python
keep_cols=X_d.select_dtypes(include=['number']).columns
X_d=X_d[keep_cols]
```

Finally, we need to get our X_train and X_test back


```python
if len(X_d.loc['first'])==1460:
    X_train=X_d.loc['first']
    X_test=X_d.loc['second']
else:
    X_train=X_d.loc['second']
    X_test=X_d.loc['first']
```

### deal with missing data
pandas provides us with a convienent way to fill missing data with average/median. Here we choose to fill the NA with average. Note to self: sometimes we use median() to avoid the influence by outlier.


```python
X_train=X_train.fillna(X_train.mean())
X_test=X_test.fillna(X_test.mean())
```

### Use StandardScaler to make data better for your model
There are some methods to scale data in scikit, like standardScaler, RobustScaler. Here we choose StandardScaler.


```python
ss=StandardScaler()
X_scale=ss.fit_transform(X_train)
X_test_scale=ss.transform(X_test)
```

### Choose your linear model
In scikit, we have,emmmmm,let's see:
- LinearRegression
- SVM
- RandomForestRegressor
- LassoCV
- RidgeCV
- ElasticCV
- GradientBoostingRegressor  

Also, you can use XGBoost for this competition. After several attempts with these models, I find GradientBoostingRegressor has the best perfermance. 


```python
gbr=GradientBoostingRegressor(n_estimators=3000,learning_rate=0.05, 
                              max_features='sqrt')
gbr.fit(X_scale,Y_train)
predict=np.array(gbr.predict(X_test_scale))
```

### Save our prediction

Lack of knowledge about python, I don't know how to add feature names when I save them as csv. So I add 'Id' and 'SalePrice' manually afterwards.


```python
final=np.hstack((testId.reshape(-1,1),predict.reshape(-1,1)))
np.savetxt('new.csv',final,delimiter=',',fmt='%d')
```

    /Library/Frameworks/Python.framework/Versions/3.5/lib/python3.5/site-packages/ipykernel_launcher.py:1: FutureWarning: reshape is deprecated and will raise in a subsequent release. Please use .values.reshape(...) instead
      """Entry point for launching an IPython kernel.


## 3.Summary
This is just a simple sample for this competition. To get better score in this competition, we need to go deeper into the feature engineering and feature selection rather than simply selecting our model and training it. Furthermore, I think this is the most important part which deserves more focus since it will determine whether you can get to the top leaderboads in competitions. 
