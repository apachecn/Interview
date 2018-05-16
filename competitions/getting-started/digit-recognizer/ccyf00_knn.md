#!/usr/bin/python
# coding: utf-8
'''
Created on 2018-05-16
Update  on 2018-05-16
Author: ccyf00
Github: https://github.com/ccyf00/kaggle-1
'''


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
train=pd.read_csv('datasets/getting-started/digit-recognizer/input/train.csv')
test=pd.read_csv('datasets/getting-started/digit-recognizer/input/test.csv')
Y_train=train["label"]
X_train=train.drop(['label'],axis=1)
del train


pca = PCA(n_components=45)
X_train_transformed=pca.fit_transform(X_train)
X_test_transformed=pca.transform(test)
X_train_pca, X_test_pca, Y_train_pca, Y_test_pca = train_test_split(
    X_train_transformed, Y_train, test_size=0.1, random_state=13)
components = [10,15,20,25,30,35,40,45]
neighbors = [2,3,4,5,6,7]
scores = np.zeros((components[len(components)-1]+1,neighbors[len(neighbors)-1]+1))
for component in components:
    for n in neighbors:
        knn=KNeighborsClassifier(n_neighbors=n)
        knn.fit(X_train_pca[:,:component],Y_train_pca)
        score = knn.score(X_test_pca[:,:component],Y_test_pca)
        scores[component][n]=score
        print('Components=',component,'neighbors = ',n,'Score = ',score)
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_pca[:,:35],Y_train_pca)
predictLabel=knn.predict(X_test_transformed[:,:35])
Submission = pd.DataFrame({"ImageId":range(1,predictLabel.shape[0]+1),
                           "Label":predictLabel})
Submission.to_csv("datasets/getting-started/digit-recognizer/ouput/KnnMnistSubmission.csv", index=False)
#分数：0.97385
