from LinearRegression import DataProcessing
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.svm import SVR


def main():
    train_path = '../clean_data/train_FD001.csv'
    test_path = '../clean_data/test_FD001.csv'

    y_test = pd.read_csv('../CMAPSSData/RUL_FD001.txt', names=['RUL'])

    drop_columns = ['settings 1', 'settings 2', 'settings 3', 'sensor 1'
                    , 'sensor 5', 'sensor 6', 'sensor 10', 'sensor 16'
                    , 'sensor 18', 'sensor 19']

    data_processor = DataProcessing(train_path, test_path)
    X_train, y_train, X_test = data_processor.prepareData(drop_columns)

    poly = PolynomialFeatures(3)
    X_train = poly.fit_transform(X_train)
    X_test = poly.fit_transform(X_test)

    svm = SVR(kernel='rbf', C=100, epsilon=0.02)
    svm.fit(X_train, y_train)

    y_train_pred = svm.predict(X_train)
    y_test_pred = svm.predict(X_test)

    data_processor.evaluate(y_train, y_train_pred)
    data_processor.evaluate(y_test, y_test_pred)


if __name__ == '__main__':
    main()

