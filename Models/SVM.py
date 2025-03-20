from LinearRegression import prepareTrainData, evaluate
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.svm import SVR


train = pd.read_csv('../clean_data/train_FD001.csv')
test = pd.read_csv('../clean_data/test_FD001.csv')
y_test = pd.read_csv('../CMAPSSData/RUL_FD001.txt', names=['RUL'])

drop_coloumns = ['settings 1', 'settings 2', 'settings 3', 'sensor 1', 'sensor 5', 'sensor 6', 'sensor 10',
                 'sensor 16', 'sensor 18', 'sensor 19']


X_train, y_train = prepareTrainData(train, drop_coloumns)
X_test = test.groupby('unit number').last().reset_index()
X_test, _ = prepareTrainData(X_test, drop_coloumns)

poly = PolynomialFeatures(3)
X_train = poly.fit_transform(X_train)
X_test = poly.fit_transform(X_test)

svm = SVR(kernel='rbf', C=100, epsilon=0.02)
svm.fit(X_train, y_train)

y_train_pred = svm.predict(X_train)
y_test_pred = svm.predict(X_test)

evaluate(y_train, y_train_pred)
evaluate(y_test, y_test_pred)


