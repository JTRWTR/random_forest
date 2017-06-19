# coding = utf-8

import numpy as np
import pandas as pd
from sklearn.datasets import load_boston

from sklearn.cross_validation import train_test_split

from sklearn.ensemble import RandomForestRegressor


from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# データをロード
boston = load_boston()
df = pd.DataFrame(boston.data, columns = boston.feature_names)
df['MEDV'] = np.array(boston.target)
# print(df)

# 説明変数、目的変数
X = df.iloc[:, :-1].values
y = df.loc[:, 'MEDV'].values

# print(X, y)

# trainデータ　と　testデータ　に　分解
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size = 0.3, random_state = 666)

forest = RandomForestRegressor()
forest.fit(X_train, y_train)

# 予測値を計算
y_train_pred = forest.predict(X_train)
y_test_pred = forest.predict(X_test)
# MSEの計算
print('MSE train : %.3f, test : %.3f' % (mean_squared_error(y_train, y_train_pred), mean_squared_error(y_test, y_test_pred)) )
# R^2の計算
print('R^2 train : %.3f, test : %.3f' % (r2_score(y_train, y_train_pred), r2_score(y_test, y_test_pred)) )
