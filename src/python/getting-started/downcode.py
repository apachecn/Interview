import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor


train_df = pd.read_csv("train.csv",index_col = 0)
test_df = pd.read_csv('test.csv',index_col = 0)

prices = pd.DataFrame({'price':train_df['SalePrice'],'log(price+1)':np.log1p(train_df['SalePrice'])})

y_train = np.log1p(train_df.pop('SalePrice'))
all_df = pd.concat((train_df,test_df),axis = 0)



all_df['MSSubClass'] = all_df['MSSubClass'].astype(str)

all_dummy_df = pd.get_dummies(all_df)

mean_cols = all_dummy_df.mean()

all_dummy_df = all_dummy_df.fillna(mean_cols)


numeric_cols = all_df.columns[all_df.dtypes != 'object']
numeric_col_means = all_dummy_df.loc[:,numeric_cols].mean()
numeric_col_std = all_dummy_df.loc[:,numeric_cols].std()
all_dummy_df.loc[:,numeric_cols] = (all_dummy_df.loc[:,numeric_cols] - numeric_col_means) / numeric_col_std


dummy_train_df = all_dummy_df.loc[train_df.index]
dummy_test_df = all_dummy_df.loc[test_df.index]


X_train = dummy_train_df.values
X_test = dummy_test_df.values




ridge = Ridge(alpha = 15)
rf = RandomForestRegressor(n_estimators = 500,max_features = .3)
ridge.fit(X_train,y_train)
rf.fit(X_train,y_train)

y_ridge = np.expm1(ridge.predict(X_test))
y_rf = np.expm1(rf.predict(X_test))

y_final = (y_ridge + y_rf) / 2


submission_df = pd.DataFrame(data = {'Id':test_df.index,'SalePrice':y_final})

submission_df.to_csv('submission.csv',columns = ['Id','SalePrice'],index = False)

