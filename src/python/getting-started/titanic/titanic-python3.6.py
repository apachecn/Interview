#!/usr/bin/python
# coding: utf-8
'''
Created on 2019-08-14
Update  on 2019-08-14
Author: 片刻
Github: https://github.com/apachecn/Interview
'''
import re
import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier


# 加载数据
def opencsv():
    root_path = '/opt/data/kaggle/getting-started/titanic'

    tr_data = pd.read_csv('%s/%s' % (root_path, 'input/train.csv'), header=0)
    te_data = pd.read_csv('%s/%s' % (root_path, 'input/test.csv'), header=0)
        
    # print(tr_data.head(5))
    # print(tr_data.info())
    # 返回数值型变量的统计量
    # print(tr_data.describe())

    # 数据预处理（清洗、缺失值）
    do_DataPreprocessing(tr_data)
    print(tr_data.head(5))
    print(tr_data.dtypes)
    print(te_data.describe())

    do_DataPreprocessing(te_data)
    print(te_data.head(5))
    print(te_data.dtypes)
    print(te_data.describe())

    # # 相关性分析
    # # 相关性协方差表, corr()函数,返回结果接近0说明无相关性,大于0说明是正相关,小于0是负相关.
    # train_corr = tr_data.corr()
    # print(train_corr)
    # # 画出相关性热力图
    # a = plt.subplots(figsize=(15,9))#调整画布大小
    # a = sns.heatmap(train_corr, vmin=-1, vmax=1 , annot=True , square=True)  #画热力图
    # plt.show()
    # """
    #             Survived    Pclass       Sex       Age     Parch      Fare  Embarked     Title  NameLength
    # Survived    1.000000 -0.338481  0.543351 -0.064910  0.081629  0.257307  0.106811  0.354072    0.332350
    # Pclass     -0.338481  1.000000 -0.131900 -0.339898  0.018443 -0.549500  0.045702 -0.211552   -0.220001
    # Sex         0.543351 -0.131900  1.000000 -0.081163  0.245489  0.182333  0.116569  0.419760    0.448759
    # Age        -0.064910 -0.339898 -0.081163  1.000000 -0.172482  0.096688 -0.009165 -0.037174    0.039702
    # Parch       0.081629  0.018443  0.245489 -0.172482  1.000000  0.216225 -0.078665  0.235164    0.252282
    # Fare        0.257307 -0.549500  0.182333  0.096688  0.216225  1.000000  0.062142  0.122872    0.155832
    # Embarked    0.106811  0.045702  0.116569 -0.009165 -0.078665  0.062142  1.000000  0.055788   -0.107749
    # Title       0.354072 -0.211552  0.419760 -0.037174  0.235164  0.122872  0.055788  1.000000    0.436099
    # NameLength  0.332350 -0.220001  0.448759  0.039702  0.252282  0.155832 -0.107749  0.436099    1.000000
    # """

    # 对于PessengerId 忽略，这个是自增长没意义    
    pids = te_data['PassengerId'].tolist()
    tr_data.drop(['PassengerId'], axis=1,inplace=True)
    te_data.drop(['PassengerId'], axis=1,inplace=True)
    train_data  = tr_data.values[:, 1:]  # 读入全部训练数据,  [行，列]
    train_label = tr_data.values[:, 0]  # 读取列表的第一列
    test_data   = te_data.values[:, :]  # 测试全部测试个数据
    return train_data, train_label, test_data, pids


def do_DataPreprocessing(titanic):
    """
    | Survival    | 生存                | 0 = No, 1 = Yes |
    | Pclass      | 票类别-社会地位       | 1 = 1st, 2 = 2nd, 3 = 3rd |  
    | Name        | 姓名                | |
    | Sex         | 性别                | |
    | Age         | 年龄                | |    
    | SibSp       | 船上的兄弟姐妹/配偶   | | 
    | Parch       | 船上的父母/孩子的数量 | |
    | Ticket      | 票号                | |   
    | Fare        | 乘客票价            | |  
    | Cabin       | 客舱号码            | |    
    | Embarked    | 登船港口            | C=Cherbourg, Q=Queenstown, S=Southampton |  

    >>> print(titanic.describe())
           PassengerId    Survived      Pclass         Age       SibSp       Parch        Fare
    count   891.000000  891.000000  891.000000  714.000000  891.000000  891.000000  891.000000
    mean    446.000000    0.383838    2.308642   29.699118    0.523008    0.381594   32.204208
    std     257.353842    0.486592    0.836071   14.526497    1.102743    0.806057   49.693429
    min       1.000000    0.000000    1.000000    0.420000    0.000000    0.000000    0.000000
    25%     223.500000    0.000000    2.000000   20.125000    0.000000    0.000000    7.910400
    50%     446.000000    0.000000    3.000000   28.000000    0.000000    0.000000   14.454200
    75%     668.500000    1.000000    3.000000   38.000000    1.000000    0.000000   31.000000
    max     891.000000    1.000000    3.000000   80.000000    8.000000    6.000000  512.329200

    Pclass  Name                          Sex     Age      SibSp   Parch   Ticket      Fare        Cabin   Embarked
    3       Braund, Mr. Owen Harris       male    22       1       0       A/5 21171   7.25                S
    1       Cumings, Mrs. John Bradley    female  38       1       0       PC 17599    71.2833     C85     C
    """
    # 组合特征(特征组合相关性变差了)
    # titanic["FamilySize"] = titanic["SibSp"] + titanic["Parch"]

    # 对缺失值处理（Age 中位数不错）
    titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())
    titanic["Fare"] = titanic["Fare"].fillna(titanic["Fare"].median())

    # 对文本特征进行处理（性别， 登船港口）
    print(titanic["Sex"].unique())
    titanic.loc[titanic["Sex"]=="male", "Sex"] = 0
    titanic.loc[titanic["Sex"]=="female", "Sex"] = 1

    # S的概率最大，当然我们也可以按照概率随机算，都可以
    print(titanic["Embarked"].unique())
    """
    titanic[["Embarked"]].groupby("Embarked").agg({"Embarked": "count"})
              Embarked
    Embarked          
    C              168
    Q               77
    S              644
    """
    titanic["Embarked"] = titanic["Embarked"].fillna('S')
    titanic.loc[titanic["Embarked"] == "S", "Embarked"] = 0
    titanic.loc[titanic["Embarked"] == "C", "Embarked"] = 1
    titanic.loc[titanic["Embarked"] == "Q", "Embarked"] = 2

    def get_title(name):
        # 名字的尊称
        title_search = re.search(' ([A-Za-z]+)\.', name)
        if title_search:
            return title_search.group(1)
        return ""
    titles = titanic["Name"].apply(get_title)
    # print(pandas.value_counts(titles))
    # 对尊称建立mapping字典
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8, "Mme": 8, "Don": 9, "Dona": 9, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2}
    for k, v in title_mapping.items():
        titles[titles == k] = v
    # print(pd.value_counts(titles))

    # 添加一个新特征表示拥护尊称
    titanic["Title"] = [int(i) for i in titles.values.tolist()]
    # 添加一个新特征表示名字长度
    titanic["NameLength"] = titanic["Name"].apply(lambda x: len(x))

    # 相关性太差，删除
    # titanic.drop(['PassengerId'], axis=1,inplace=True)
    titanic.drop(['Cabin'], axis=1,inplace=True)
    titanic.drop(['SibSp'], axis=1,inplace=True)
    # titanic.drop(['Parch'],axis=1,inplace=True)
    titanic.drop(['Ticket'], axis=1,inplace=True)
    titanic.drop(['Name'],   axis=1,inplace=True)


def do_FeatureEngineering(data, COMPONENT_NUM=0.9):
    # scale values  对一化
    scaler = preprocessing.StandardScaler()
    s_data = scaler.fit_transform(data)
    return s_data

    # # 降维(不降维，准确率还上升了)
    # '''
    # 使用说明：https://www.cnblogs.com/pinard/p/6243025.html
    # n_components>=1
    #   n_components=NUM   设置占特征数量比
    # 0 < n_components < 1
    #   n_components=0.99  设置阈值总方差占比
    # '''
    # pca = PCA(n_components=COMPONENT_NUM, whiten=False)
    # pca.fit(s_data)  # Fit the model with X
    # pca_data = pca.transform(s_data)  # Fit the model with X and 在X上完成降维.

    # # pca 方差大小、方差占比、特征数量
    # # print("方差大小:\n", pca.explained_variance_, "方差占比:\n", pca.explained_variance_ratio_)
    # print("特征数量: %s" % pca.n_components_)
    # print("总方差占比: %s" % sum(pca.explained_variance_ratio_))

    # return pca_data


def trainModel(trainData, trainLabel):

    # 模拟测试
    # # 0.8069524400247253 [0.79329609 0.81564246 0.8258427  0.80337079 0.79661017]
    # # model = LogisticRegression(random_state=1)
    # # 0.8272091118939124 [0.82122905 0.80446927 0.84831461 0.82022472 0.84180791]
    # model = RandomForestClassifier(random_state=1, n_estimators=100, min_samples_split=4, min_samples_leaf=2)
    # # 0.822670577600365  [0.82681564 0.82122905 0.83146067 0.80898876 0.82485876]
    # # model = RandomForestClassifier(random_state=1, n_estimators=50, min_samples_split=8, min_samples_leaf=4)

   #  # 交叉验证部分 #####
   #  from sklearn.model_selection import GridSearchCV
   #  param_test = {
   #      # 'n_estimators': np.arange(190, 240, 2), 
   #      # 'max_depth': np.arange(4, 7, 1), 
   #      # 'learning_rate': np.array([0.01, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2]), 

   #      'n_estimators': np.array([196]), 
   #      'max_depth': np.array([4]),     
   #      'learning_rate': np.array([0.01, 0.02, 0.03, 0.04, 0.05]), 
   #      # 'min_child_weight': np.arange(1, 6, 2), 
   #      # 'C': (1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9)
   #  }

   # # 0.8294499549079417   [0.82122905 0.80446927 0.86516854 0.82022472 0.83615819]
   #  model = XGBClassifier()
   #  grid_search = GridSearchCV(estimator=model, param_grid=param_test, scoring='roc_auc', cv=5)
   #  grid_search.fit(trainData, trainLabel)
   #  print("最优得分 >>>", grid_search.best_score_)
   #  print("最优参数 >>>", grid_search.best_params_)

    # 0.8685305085155506  [0.85770751 0.82002635 0.89632353 0.87018717 0.89840799]
    model = XGBClassifier(learning_rate = 0.1,  n_estimators= 202, max_depth= 4,
                min_child_weight= 5, gamma=0, subsample=0.8, colsample_bytree=0.8, 
                objective= 'binary:logistic', scale_pos_weight=1
    )
    model = XGBClassifier(n_estimators=196, max_depth=4, learning_rate=0.03)
    scores = cross_val_score(model, trainData, trainLabel, cv=5, scoring='roc_auc')
    print(scores.mean(), "\n", scores)

    print("模型融合")
    """
    Bagging:   同一模型的投票选举
    Boosting:  同一模型的再学习
    Voting:    不同模型的投票选举
    Stacking:  分层预测 – K-1份数据预测1份模型拼接，对结果在进行预测
    Blending:  分层预测 – 将数据分成2部分，A部分训练B部分得到预测结果，得到 预测结果*算法数 => 从而预测最终结果
    """
    # 1. Bagging 算法实现
    # 0.8691726623564537  [0.86179183 0.82700922 0.8855615  0.87700535 0.89449541]
    # model = RandomForestClassifier(random_state=1, n_estimators=100, min_samples_split=4, min_samples_leaf=2)

    # 2. Boosting 算法实现
    # 0.8488710896477386  [0.8198946  0.82285903 0.87780749 0.84906417 0.87473017]
    # model = AdaBoostClassifier(random_state=1, n_estimators=100, learning_rate=1)

    # # 3. Voting
    # # 0.8695399796790022  [0.87259552 0.8370224  0.87433155 0.86885027 0.89490016]
    # model = VotingClassifier(
    #     estimators=[
    #         ('log_clf', LogisticRegression()),
    #         ('ab_clf', AdaBoostClassifier()),
    #         ('svm_clf', SVC(probability=True)),
    #         ('rf_clf', RandomForestClassifier()),
    #         ('gbdt_clf', GradientBoostingClassifier()),
    #         ('rb_clf', AdaBoostClassifier())
    #     ], voting='soft') # , voting='hard')
    # scores = cross_val_score(model, trainData, trainLabel, cv=5, scoring='roc_auc')
    # print(scores.mean(), "\n", scores)
    
    # # 4. Stacking
    # # 0.8713813265814722  [0.87747036 0.83886693 0.86590909 0.87085561 0.90380464]
    # clfs = [
    #     AdaBoostClassifier(),
    #     SVC(probability=True),
    #     AdaBoostClassifier(),
    #     LogisticRegression(C=0.1,max_iter=100),
    #     XGBClassifier(max_depth=6,n_estimators=100,num_round = 5),
    #     RandomForestClassifier(n_estimators=100,max_depth=6,oob_score=True),
    #     GradientBoostingClassifier(learning_rate=0.3,max_depth=6,n_estimators=100)
    # ]

    # # from sklearn.cross_validation import StratifiedKFold
    # # n_folds = 5
    # # skf = list(StratifiedKFold(trainLabel, n_folds))
    # kf = KFold(n_splits=5, shuffle=True, random_state=1)

    # # 创建零矩阵
    # dataset_stacking_train = np.zeros((trainData.shape[0], len(clfs)))
    # # dataset_stacking_label  = np.zeros((trainLabel.shape[0], len(clfs)))

    # for j, clf in enumerate(clfs):
    #     '''依次训练各个单模型'''
    #     for i,(train, test) in enumerate(kf.split(trainLabel)):
    #         '''使用第i个部分作为预测，剩余的部分来训练模型，获得其预测的输出作为第i部分的新特征。'''
    #         # print("Fold", i)
    #         X_train, y_train, X_test, y_test = trainData[train], trainLabel[train], trainData[test], trainLabel[test]
    #         clf.fit(X_train, y_train)
    #         y_submission = clf.predict_proba(X_test)[:, 1]

    #         # j 表示每一次的算法，而 test是交叉验证得到的每一行（也就是每一个算法把测试机和都预测了一遍）
    #         dataset_stacking_train[test, j] = y_submission
     
    # # 用建立第二层模型
    # model = LogisticRegression(C=0.1, max_iter=100)
    # model.fit(dataset_stacking_train, trainLabel)

    # scores = cross_val_score(model, dataset_stacking_train, trainLabel, cv=5, scoring='roc_auc')
    # print(scores.mean(), "\n", scores)
    
    # 5. Blending
    # 0.8838950287185581 [0.87584416 0.91064935 0.89714286 0.85294118 0.8828976 ]
    clfs = [
        AdaBoostClassifier(),
        SVC(probability=True),
        AdaBoostClassifier(),
        LogisticRegression(C=0.1,max_iter=100),
        XGBClassifier(max_depth=6,n_estimators=100,num_round = 5),
        RandomForestClassifier(n_estimators=100,max_depth=6,oob_score=True),
        GradientBoostingClassifier(learning_rate=0.3,max_depth=6,n_estimators=100)
    ]
    X_d1, X_d2, y_d1, y_d2 = train_test_split(trainData, trainLabel, test_size=0.5, random_state=2017)
    dataset_d1 = np.zeros((X_d2.shape[0], len(clfs)))
    dataset_d2 = np.zeros((trainLabel.shape[0], len(clfs)))

    for j, clf in enumerate(clfs):
        #依次训练各个单模型
        # 对于测试集，直接用这k个模型的预测值作为新的特征。
        clf.fit(X_d1, y_d1)
        dataset_d1[:, j] = clf.predict_proba(X_d2)[:, 1]

    # 用建立第二层模型
    model = LogisticRegression(C=0.1, max_iter=100)
    model.fit(dataset_d1, y_d2)

    scores = cross_val_score(model, dataset_d1, y_d2, cv=5, scoring='roc_auc')
    print(scores.mean(), "\n", scores)
    
    return model


def main():
    # 开始时间
    sta_time = datetime.datetime.now()

    # 1.加载数据和预处理
    train_data, train_label, test_data, pids = opencsv()

    # 特征工程
    pca_tr_data = do_FeatureEngineering(train_data)
    pca_te_data = do_FeatureEngineering(test_data)

    # 模型训练（分类问题： lr、rf、adboost、xgboost、lightgbm）
    model = trainModel(pca_tr_data, train_label)
    model.fit(pca_tr_data, train_label)
    labels = model.predict(pca_te_data)

    print(type(pids), type(labels.tolist()))
    result = pd.DataFrame({
        'PassengerId': pids, 
        'Survived': labels
    })
    result.to_csv('Result_titanic.csv', index=False)

    # 结束时间
    end_time = datetime.datetime.now()
    times = (end_time - sta_time).seconds
    print("\n运行时间: %ss == %sm == %sh\n\n" % (times, times/60, times/60/60))


if __name__ == "__main__":
    main()
