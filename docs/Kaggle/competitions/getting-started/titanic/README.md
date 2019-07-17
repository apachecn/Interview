# **泰坦尼克号**

## 比赛说明

* [**泰坦尼克号**](https://www.kaggle.com/c/titanic) 的沉没是历史上最臭名昭着的沉船之一。1912年4月15日，在首航期间，泰坦尼克号撞上一座冰山后沉没，2224名乘客和机组人员中有1502人遇难。这一耸人听闻的悲剧震撼了国际社会，并导致了更好的船舶安全条例。
* 沉船导致生命损失的原因之一是乘客和船员没有足够的救生艇。虽然幸存下来的运气有一些因素，但有些人比其他人更有可能生存，比如妇女，儿童和上层阶级。
* 在这个挑战中，我们要求你完成对什么样的人可能生存的分析。特别是，我们要求你运用机器学习的工具来预测哪些乘客幸存下来的悲剧。

## 参赛成员

* 开源组织: [ApacheCN ~ apachecn.org](http://www.apachecn.org/)
* **菜鸡互啄(团队)**: [@那伊抹微笑](https://github.com/wangyangting), [@风风](https://github.com/fengfengtzp), [@李铭哲](https://github.com/limingzhe), [@刘海飞](https://github.com/WindZQ), [@DL - 小王子](https://github.com/VPrincekin), [@成飘飘](https://github.com/chengpiaopiao)
* **瑶瑶亲卫队(团队)**: [@瑶妹](https://github.com/chenyyx) 张俊皓 kngines [@谈笑风生](https://github.com/zhu1040028623) Yukine ~守护 程威 屋檐听雨 Gladiator 吃着狗粮的电酱PRPR

## 比赛分析

* 分类问题：预测的是`生`与`死`的问题
* 常用算法： `K紧邻(knn)`、`逻辑回归(LogisticRegression)`、`随机森林(RandomForest)`、`支持向量机(SVM)`、`xgboost`、`GBDT`

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

* 数据集下载地址：<https://www.kaggle.com/c/titanic/data>

```python
# 导入相关数据包
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
```

```python
root_path = '/opt/data/datasets/getting-started/titanic/input'

train = pd.read_csv('%s/%s' % (root_path, 'train.csv'))
test = pd.read_csv('%s/%s' % (root_path, 'test.csv'))
```

### 特征说明

| 特征 | 描述 | 值|
| - | - | - |
| survival | 生存           | 0 = No, 1 = Yes |
| pclass   | 票类别-社会地位  | 1 = 1st, 2 = 2nd, 3 = 3rd |  
| sex      | 性别           | |
| Age      | 年龄           | |    
| sibsp    | 兄弟姐妹/配偶   | | 
| parch    | 父母/孩子的数量 | |
| ticket   | 票号           | |   
| fare     | 乘客票价       | |  
| cabin    | 客舱号码       | |    
| embarked | 登船港口       | C=Cherbourg, Q=Queenstown, S=Southampton |  

### 特征详情

```python
train.head(5)
```

![](/img/competitions/getting-started/titanic/titanic_top_5.jpg)

```python
# 返回每列列名,该列非nan值个数,以及该列类型
train.info()

<class 'pandas.core.frame.DataFrame'>
RangeIndex: 891 entries, 0 to 890
Data columns (total 12 columns):
PassengerId    891 non-null int64
Survived       891 non-null int64
Pclass         891 non-null int64
Name           891 non-null object
Sex            891 non-null object
Age            714 non-null float64
SibSp          891 non-null int64
Parch          891 non-null int64
Ticket         891 non-null object
Fare           891 non-null float64
Cabin          204 non-null object
Embarked       889 non-null object
dtypes: float64(2), int64(5), object(5)
memory usage: 83.6+ KB
```

```python
test.info()

<class 'pandas.core.frame.DataFrame'>
RangeIndex: 418 entries, 0 to 417
Data columns (total 11 columns):
PassengerId    418 non-null int64
Pclass         418 non-null int64
Name           418 non-null object
Sex            418 non-null object
Age            332 non-null float64
SibSp          418 non-null int64
Parch          418 non-null int64
Ticket         418 non-null object
Fare           417 non-null float64
Cabin          91 non-null object
Embarked       418 non-null object
dtypes: float64(2), int64(4), object(5)
memory usage: 36.0+ KB
```

```python
# 返回数值型变量的统计量
# train.describe(percentiles=[0.00, 0.25, 0.5, 0.75, 1.00])
train.describe()
```

![](/img/competitions/getting-started/titanic/titanic_train_desc.jpg)

### 特征分析（统计学与绘图）
目的:初步了解数据之间的相关性,为构造特征工程以及模型建立做准备


```python
# 存活人数
train['Survived'].value_counts()

0    549
1    342
Name: Survived, dtype: int64
```

> 1)数值型数据协方差,corr()函数

来个总览,快速了解个数据的相关性

```python
# 相关性协方差表,corr()函数,返回结果接近0说明无相关性,大于0说明是正相关,小于0是负相关.
train_corr = train.drop('PassengerId',axis=1).corr()
train_corr
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Survived</th>
      <td>1.000000</td>
      <td>-0.338481</td>
      <td>-0.077221</td>
      <td>-0.035322</td>
      <td>0.081629</td>
      <td>0.257307</td>
    </tr>
    <tr>
      <th>Pclass</th>
      <td>-0.338481</td>
      <td>1.000000</td>
      <td>-0.369226</td>
      <td>0.083081</td>
      <td>0.018443</td>
      <td>-0.549500</td>
    </tr>
    <tr>
      <th>Age</th>
      <td>-0.077221</td>
      <td>-0.369226</td>
      <td>1.000000</td>
      <td>-0.308247</td>
      <td>-0.189119</td>
      <td>0.096067</td>
    </tr>
    <tr>
      <th>SibSp</th>
      <td>-0.035322</td>
      <td>0.083081</td>
      <td>-0.308247</td>
      <td>1.000000</td>
      <td>0.414838</td>
      <td>0.159651</td>
    </tr>
    <tr>
      <th>Parch</th>
      <td>0.081629</td>
      <td>0.018443</td>
      <td>-0.189119</td>
      <td>0.414838</td>
      <td>1.000000</td>
      <td>0.216225</td>
    </tr>
    <tr>
      <th>Fare</th>
      <td>0.257307</td>
      <td>-0.549500</td>
      <td>0.096067</td>
      <td>0.159651</td>
      <td>0.216225</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>

```python
# 画出相关性热力图
a = plt.subplots(figsize=(15,9))#调整画布大小
a = sns.heatmap(train_corr, vmin=-1, vmax=1 , annot=True , square=True)#画热力图
```

![png](/img/competitions/getting-started/titanic/titanic_corr_analysis.png)

> 2)各个数据与结果的关系

进一步探索分析各个数据与结果的关系

* ① Pclass,乘客等级,1是最高级

结果分析:可以看出Survived和Pclass在Pclass=1的时候有较强的相关性（>0.5），所以最终模型中包含该特征。

```python
train.groupby(['Pclass'])['Pclass','Survived'].mean()
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pclass</th>
      <th>Survived</th>
    </tr>
    <tr>
      <th>Pclass</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>0.629630</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2.0</td>
      <td>0.472826</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3.0</td>
      <td>0.242363</td>
    </tr>
  </tbody>
</table>
</div>


```python
train[['Pclass','Survived']].groupby(['Pclass']).mean().plot.bar()

<matplotlib.axes._subplots.AxesSubplot at 0xc33fa90>
```

![png](/img/competitions/getting-started/titanic/titanic_pclass_bar.png)

* ② Sex,性别

结果分析:女性有更高的活下来的概率（74%）,保留该特征

```python
train.groupby(['Sex'])['Sex','Survived'].mean()
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
    </tr>
    <tr>
      <th>Sex</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>female</th>
      <td>0.742038</td>
    </tr>
    <tr>
      <th>male</th>
      <td>0.188908</td>
    </tr>
  </tbody>
</table>
</div>

```python
train[['Sex','Survived']].groupby(['Sex']).mean().plot.bar()

<matplotlib.axes._subplots.AxesSubplot at 0x105c4b630>
```

![png](/img/competitions/getting-started/titanic/titanic_sex_bar.png)

* ③ SibSp and Parch  兄妹配偶数/父母子女数

结果分析:这些特征与特定的值没有相关性不明显，最好是由这些独立的特征派生出一个新特征或者一组新特征

```python
train[['SibSp','Survived']].groupby(['SibSp']).mean()
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
    </tr>
    <tr>
      <th>SibSp</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.345395</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.535885</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.464286</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.250000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.166667</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>

```python
train[['Parch','Survived']].groupby(['Parch']).mean().plot.bar()

<matplotlib.axes._subplots.AxesSubplot at 0x1144385c0>
```

![png](/img/competitions/getting-started/titanic/titanic_parch_bar.png)

* ④ Age年龄与生存情况的分析.

结果分析:由图,可以看到年龄是影响生存情况的. 

但是年龄是有大部分缺失值的,缺失值需要进行处理,可以使用填充或者模型预测.

```python
g = sns.FacetGrid(train, col='Survived',size=5)
g.map(plt.hist, 'Age', bins=40)

<seaborn.axisgrid.FacetGrid at 0xc5f7cf8>
```

![png](/img/competitions/getting-started/titanic/titanic_age_map.png)

```python
train.groupby(['Age'])['Survived'].mean().plot()

<matplotlib.axes._subplots.AxesSubplot at 0xc71ac18>
``` 

![png](/img/competitions/getting-started/titanic/titanic_age_axes.png)

* ⑤ Embarked登港港口与生存情况的分析

结果分析:C地的生存率更高,这个也应该保留为模型特征.

```python
sns.countplot('Embarked',hue='Survived',data=train)

<matplotlib.axes._subplots.AxesSubplot at 0xca1e5f8>
```

![png](/img/competitions/getting-started/titanic/titanic_embarked_count.png)

* ⑥ 其他因素

在数据的Name项中包含了对该乘客的称呼，如Mr、Miss等，这些信息包含了乘客的年龄、性别、也有可能包含社会地位，如Dr、Lady、Major、Master等称呼。这一项不方便用图表展示，但是在特征工程中，我们会将其提取出来,然后放到模型中。

剩余因素还有船票价格、船舱号和船票号，这三个因素都可能会影响乘客在船中的位置从而影响逃生顺序，但是因为这三个因素与生存之间看不出明显规律，所以在后期模型融合时，将这些因素交给模型来决定其重要性。

## 二. 特征工程

```python
#先将数据集合并,一起做特征工程(注意,标准化的时候需要分开处理)
#先将test补齐,然后通过pd.apped()合并
test['Survived'] = 0
train_test = train.append(test)
```

### 特征处理

* ① Pclass,乘客等级,1是最高级

两种方式:一是该特征不做处理,可以直接保留.二是再处理:也进行分列处理(比较那种方式模型效果更好,就选那种)

```python
train_test = pd.get_dummies(train_test,columns=['Pclass'])
```

* ② Sex,性别¶
无缺失值,直接分列

```python
train_test = pd.get_dummies(train_test,columns=["Sex"])
```

* ③ SibSp and Parch  兄妹配偶数/父母子女数

第一次直接保留:这两个都影响生存率,且都是数值型,先直接保存.

第二次进行两项求和,并进行分列处理.(兄妹配偶数和父母子女数都是认识人的数量,所以总数可能也会更好)(模型结果提高到了)

```python
#这是剑豪模型后回来添加的新特征,模型的分数最终有所提高了.
train_test['SibSp_Parch'] = train_test['SibSp'] + train_test['Parch']
```

```python
train_test = pd.get_dummies(train_test,columns = ['SibSp','Parch','SibSp_Parch']) 
```

* ④ Embarked 
数据有极少量(3个)缺失值,但是在分列的时候,缺失值的所有列可以均为0,所以可以考虑不填充.

另外,也可以考虑用测试集众数来填充.先找出众数,再采用df.fillna()方法

```python
train_test = pd.get_dummies(train_test,columns=["Embarked"])
```

* ⑤ Name

1.在数据的Name项中包含了对该乘客的称呼,将这些关键词提取出来,然后做分列处理.(参考别人的)

```python
#从名字中提取出称呼： df['Name].str.extract()是提取函数,配合正则一起使用
train_test['Name1'] = train_test['Name'].str.extract('.+,(.+)', expand=False).str.extract('^(.+?)\.', expand=False).str.strip()
```

```python
#将姓名分类处理()
train_test['Name1'].replace(['Capt', 'Col', 'Major', 'Dr', 'Rev'], 'Officer' , inplace = True)
train_test['Name1'].replace(['Jonkheer', 'Don', 'Sir', 'the Countess', 'Dona', 'Lady'], 'Royalty' , inplace = True)
train_test['Name1'].replace(['Mme', 'Ms', 'Mrs'], 'Mrs')
train_test['Name1'].replace(['Mlle', 'Miss'], 'Miss')
train_test['Name1'].replace(['Mr'], 'Mr' , inplace = True)
train_test['Name1'].replace(['Master'], 'Master' , inplace = True)
```

```python
#分列处理
train_test = pd.get_dummies(train_test,columns=['Name1'])
```

2. 从姓名中提取出姓做特征

```python
#从姓名中提取出姓
train_test['Name2'] = train_test['Name'].apply(lambda x: x.split('.')[1])

#计算数量,然后合并数据集
Name2_sum = train_test['Name2'].value_counts().reset_index()
Name2_sum.columns=['Name2','Name2_sum']
train_test = pd.merge(train_test,Name2_sum,how='left',on='Name2')

#由于出现一次时该特征时无效特征,用one来代替出现一次的姓
train_test.loc[train_test['Name2_sum'] == 1 , 'Name2_new'] = 'one'
train_test.loc[train_test['Name2_sum'] > 1 , 'Name2_new'] = train_test['Name2']
del train_test['Name2']

#分列处理
train_test = pd.get_dummies(train_test,columns=['Name2_new'])
```

```python
#删掉姓名这个特征
del train_test['Name']
```

* ⑥ Fare

该特征有缺失值,先找出缺失值的那调数据,然后用平均数填充

```python
#从上面的分析,发现该特征train集无miss值,test有一个缺失值,先查看
train_test.loc[train_test["Fare"].isnull()]
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Cabin</th>
      <th>Fare</th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Ticket</th>
      <th>Pclass_1</th>
      <th>Pclass_2</th>
      <th>Pclass_3</th>
      <th>Sex_female</th>
      <th>...</th>
      <th>Name2_new_ Thomas Henry</th>
      <th>Name2_new_ Victor</th>
      <th>Name2_new_ Washington</th>
      <th>Name2_new_ William</th>
      <th>Name2_new_ William Edward</th>
      <th>Name2_new_ William Henry</th>
      <th>Name2_new_ William James</th>
      <th>Name2_new_ William John</th>
      <th>Name2_new_ William Thomas</th>
      <th>Name2_new_one</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1043</th>
      <td>60.5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1044</td>
      <td>0</td>
      <td>3701</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>1 rows × 137 columns</p>
</div>

```python
#票价与pclass和Embarked有关,所以用train分组后的平均数填充
train.groupby(by=["Pclass","Embarked"]).Fare.mean()

Pclass  Embarked
1       C           104.718529
        Q            90.000000
        S            70.364862
2       C            25.358335
        Q            12.350000
        S            20.327439
3       C            11.214083
        Q            11.183393
        S            14.644083
Name: Fare, dtype: float64
```

```python
#用pclass=3和Embarked=S的平均数14.644083来填充
train_test["Fare"].fillna(14.435422,inplace=True)
```

* ⑦ Ticket

该列和名字做类似的处理,先提取,然后分列

```python
#将Ticket提取字符列
#str.isnumeric()  如果S中只有数字字符，则返回True，否则返回False
train_test['Ticket_Letter'] = train_test['Ticket'].str.split().str[0]
train_test['Ticket_Letter'] = train_test['Ticket_Letter'].apply(lambda x:np.nan if x.isnumeric() else x)
train_test.drop('Ticket',inplace=True,axis=1)
```

```python
#分列,此时nan值可以不做处理
train_test = pd.get_dummies(train_test,columns=['Ticket_Letter'],drop_first=True)
```

* ⑧ Age

1. 该列有大量缺失值,考虑用一个回归模型进行填充.
2. 在模型修改的时候,考虑到年龄缺失值可能影响死亡情况,用年龄是否缺失值来构造新特征

```python
"""这是模型就好后回来增加的新特征
考虑年龄缺失值可能影响死亡情况,数据表明,年龄缺失的死亡率为0.19."""
train_test.loc[train_test["Age"].isnull()]['Survived'].mean()

0.19771863117870722
```

```python
# 所以用年龄是否缺失值来构造新特征
train_test.loc[train_test["Age"].isnull() ,"age_nan"] = 1
train_test.loc[train_test["Age"].notnull() ,"age_nan"] = 0
train_test = pd.get_dummies(train_test,columns=['age_nan'])
```

利用其他组特征量，采用机器学习算法来预测Age

```python
train_test.info()

<class 'pandas.core.frame.DataFrame'>
Int64Index: 1309 entries, 0 to 1308
Columns: 187 entries, Age to age_nan_1.0
dtypes: float64(2), int64(3), object(1), uint8(181)
memory usage: 343.0+ KB
```

```python
#创建没有['Age','Survived']的数据集
missing_age = train_test.drop(['Survived','Cabin'],axis=1)
#将Age完整的项作为训练集、将Age缺失的项作为测试集。
missing_age_train = missing_age[missing_age['Age'].notnull()]
missing_age_test = missing_age[missing_age['Age'].isnull()]
```

```python
#构建训练集合预测集的X和Y值
missing_age_X_train = missing_age_train.drop(['Age'], axis=1)
missing_age_Y_train = missing_age_train['Age']
missing_age_X_test = missing_age_test.drop(['Age'], axis=1)
```

```python
# 先将数据标准化
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
#用测试集训练并标准化
ss.fit(missing_age_X_train)
missing_age_X_train = ss.transform(missing_age_X_train)
missing_age_X_test = ss.transform(missing_age_X_test)
```

```python
#使用贝叶斯预测年龄
from sklearn import linear_model
lin = linear_model.BayesianRidge()
```

```python
lin.fit(missing_age_X_train,missing_age_Y_train)

BayesianRidge(alpha_1=1e-06, alpha_2=1e-06, compute_score=False, copy_X=True,
        fit_intercept=True, lambda_1=1e-06, lambda_2=1e-06, n_iter=300,
        normalize=False, tol=0.001, verbose=False)
```

```python
#利用loc将预测值填入数据集
train_test.loc[(train_test['Age'].isnull()), 'Age'] = lin.predict(missing_age_X_test)
```

```python
#将年龄划分是个阶段10以下,10-18,18-30,30-50,50以上
train_test['Age'] = pd.cut(train_test['Age'], bins=[0,10,18,30,50,100],labels=[1,2,3,4,5])

train_test = pd.get_dummies(train_test,columns=['Age'])
```

* ⑨ Cabin

cabin项缺失太多，只能将有无Cain首字母进行分类,缺失值为一类,作为特征值进行建模,也可以考虑直接舍去该特征

```python
#cabin项缺失太多，只能将有无Cain首字母进行分类,缺失值为一类,作为特征值进行建模
train_test['Cabin_nan'] = train_test['Cabin'].apply(lambda x:str(x)[0] if pd.notnull(x) else x)
train_test = pd.get_dummies(train_test,columns=['Cabin_nan'])
```

```python
#cabin项缺失太多，只能将有无Cain首字母进行分类,
train_test.loc[train_test["Cabin"].isnull() ,"Cabin_nan"] = 1
train_test.loc[train_test["Cabin"].notnull() ,"Cabin_nan"] = 0
train_test = pd.get_dummies(train_test,columns=['Cabin_nan'])
train_test.drop('Cabin',axis=1,inplace=True)
```

* ⑩ 特征工程处理完了,划分数据集

```python
train_data = train_test[:891]
test_data = train_test[891:]
train_data_X = train_data.drop(['Survived'],axis=1)
train_data_Y = train_data['Survived']
test_data_X = test_data.drop(['Survived'],axis=1)
```

### 数据规约

1. 线性模型需要用标准化的数据建模,而树类模型不需要标准化的数据
2. 处理标准化的时候,注意将测试集的数据transform到test集上

```python
from sklearn.preprocessing import StandardScaler
ss2 = StandardScaler()
ss2.fit(train_data_X)
train_data_X_sd = ss2.transform(train_data_X)
test_data_X_sd = ss2.transform(test_data_X)
```


## 三. 建立模型

### 模型发现

1. 可选单个模型模型有随机森林,逻辑回归,svm,xgboost,gbdt等.
2. 也可以将多个模型组合起来,进行模型融合,比如voting,stacking等方法
3. 好的特征决定模型上限,好的模型和参数可以无线逼近上限.
4. 我测试了多种模型,模型结果最高的随机森林,最高有0.8.

### 构建模型

> 随机森林

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=150,min_samples_leaf=3,max_depth=6,oob_score=True)
rf.fit(train_data_X,train_data_Y)

test["Survived"] = rf.predict(test_data_X)
RF = test[['PassengerId','Survived']].set_index('PassengerId')
RF.to_csv('RF.csv')
```

```python
# 随机森林是随机选取特征进行建模的,所以每次的结果可能都有点小差异
# 如果分数足够好,可以将该模型保存起来,下次直接调出来使用0.81339 'rf10.pkl'
from sklearn.externals import joblib
joblib.dump(rf, 'rf10.pkl')
```

> LogisticRegression

```python
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV

lr = LogisticRegression()
param = {'C':[0.001,0.01,0.1,1,10], "max_iter":[100,250]}
clf = GridSearchCV(lr, param,cv=5, n_jobs=-1, verbose=1, scoring="roc_auc")
clf.fit(train_data_X_sd, train_data_Y)

# 打印参数的得分情况
clf.grid_scores_
# 打印最佳参数
clf.best_params_

# 将最佳参数传入训练模型
lr = LogisticRegression(clf.best_params_)
lr.fit(train_data_X_sd, train_data_Y)

# 输出结果
test["Survived"] = lr.predict(test_data_X_sd)
test[['PassengerId', 'Survived']].set_index('PassengerId').to_csv('LS5.csv')
```

> SVM

```python
from sklearn import svm
svc = svm.SVC()

clf = GridSearchCV(svc,param,cv=5,n_jobs=-1,verbose=1,scoring="roc_auc")
clf.fit(train_data_X_sd,train_data_Y)

clf.best_params_

svc = svm.SVC(C=1,max_iter=250)

# 训练模型并预测结果
svc.fit(train_data_X_sd,train_data_Y)
svc.predict(test_data_X_sd)

# 打印结果
test["Survived"] = svc.predict(test_data_X_sd)
SVM = test[['PassengerId','Survived']].set_index('PassengerId')
SVM.to_csv('svm1.csv')
```

> GBDT

```python
from sklearn.ensemble import GradientBoostingClassifier

gbdt = GradientBoostingClassifier(learning_rate=0.7,max_depth=6,n_estimators=100,min_samples_leaf=2)

gbdt.fit(train_data_X,train_data_Y)

test["Survived"] = gbdt.predict(test_data_X)
test[['PassengerId','Survived']].set_index('PassengerId').to_csv('gbdt3.csv')
```

> xgboost

```python
import xgboost as xgb

xgb_model = xgb.XGBClassifier(n_estimators=150,min_samples_leaf=3,max_depth=6)
xgb_model.fit(train_data_X,train_data_Y)

test["Survived"] = xgb_model.predict(test_data_X)
XGB = test[['PassengerId','Survived']].set_index('PassengerId')
XGB.to_csv('XGB5.csv')
```

## 四 建立模型

> 模型融合 voting

```python
from sklearn.ensemble import VotingClassifier

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C=0.1,max_iter=100)

import xgboost as xgb
xgb_model = xgb.XGBClassifier(max_depth=6,min_samples_leaf=2,n_estimators=100,num_round = 5)

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=200,min_samples_leaf=2,max_depth=6,oob_score=True)

from sklearn.ensemble import GradientBoostingClassifier
gbdt = GradientBoostingClassifier(learning_rate=0.1,min_samples_leaf=2,max_depth=6,n_estimators=100)

vot = VotingClassifier(estimators=[('lr', lr), ('rf', rf),('gbdt',gbdt),('xgb',xgb_model)], voting='hard')
vot.fit(train_data_X_sd,train_data_Y)

test["Survived"] = vot.predict(test_data_X_sd)
test[['PassengerId','Survived']].set_index('PassengerId').to_csv('vot5.csv')
```

> 模型融合 stacking


```python
# 划分train数据集,调用代码,把数据集名字转成和代码一样
X = train_data_X_sd
X_predict = test_data_X_sd
y = train_data_Y

'''模型融合中使用到的各个单模型'''
from sklearn.linear_model import LogisticRegression
from sklearn import svm
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

clfs = [LogisticRegression(C=0.1,max_iter=100),
        xgb.XGBClassifier(max_depth=6,n_estimators=100,num_round = 5),
        RandomForestClassifier(n_estimators=100,max_depth=6,oob_score=True),
        GradientBoostingClassifier(learning_rate=0.3,max_depth=6,n_estimators=100)]

# 创建n_folds
from sklearn.cross_validation import StratifiedKFold
n_folds = 5
skf = list(StratifiedKFold(y, n_folds))

# 创建零矩阵
dataset_blend_train = np.zeros((X.shape[0], len(clfs)))
dataset_blend_test = np.zeros((X_predict.shape[0], len(clfs)))

# 建立模型
for j, clf in enumerate(clfs):
    '''依次训练各个单模型'''
    # print(j, clf)
    dataset_blend_test_j = np.zeros((X_predict.shape[0], len(skf)))
    for i, (train, test) in enumerate(skf):
        '''使用第i个部分作为预测，剩余的部分来训练模型，获得其预测的输出作为第i部分的新特征。'''
        # print("Fold", i)
        X_train, y_train, X_test, y_test = X[train], y[train], X[test], y[test]
        clf.fit(X_train, y_train)
        y_submission = clf.predict_proba(X_test)[:, 1]
        dataset_blend_train[test, j] = y_submission
        dataset_blend_test_j[:, i] = clf.predict_proba(X_predict)[:, 1]
    '''对于测试集，直接用这k个模型的预测值均值作为新的特征。'''
    dataset_blend_test[:, j] = dataset_blend_test_j.mean(1)

# 用建立第二层模型
clf2 = LogisticRegression(C=0.1,max_iter=100)
clf2.fit(dataset_blend_train, y)
y_submission = clf2.predict_proba(dataset_blend_test)[:, 1]

test = pd.read_csv("test.csv")
test["Survived"] = clf2.predict(dataset_blend_test)
test[['PassengerId','Survived']].set_index('PassengerId').to_csv('stack3.csv')
```
