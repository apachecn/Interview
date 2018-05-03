# coding: utf-8

from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV

iris = datasets.load_iris()
parameters = {"kernel": ("linear", "rbf"), "C": range(1, 10)}
svr = svm.SVC()

'''
clf.fit(): 运行网格搜索
grid_scores_: 给出不同参数情况下的评价结果
best_params_: 描述了已取得最佳结果的参数的组合
best_score_: 成员提供优化过程期间观察到的最好的评分
'''
clf = GridSearchCV(svr, parameters)
clf.fit(iris.data, iris.target)
print(clf.grid_scores_)  # 所有情况的评价结果
print(clf.best_params_)  # 最好的参数
print(clf.best_score_)   # 最好的参数的平均得分
