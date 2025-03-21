import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from LinearRegression import DataProcessing
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

def main():
    train_path = '../clean_data/train_FD001.csv'
    test_path = '../clean_data/test_FD001.csv'

    y_test = pd.read_csv('../CMAPSSData/RUL_FD001.txt', names=['RUL'])

    drop_columns = ['settings 1', 'settings 2', 'settings 3', 'sensor 1'
                    , 'sensor 5', 'sensor 6', 'sensor 10', 'sensor 16'
                    , 'sensor 18', 'sensor 19']

    data_processor = DataProcessing(train_path, test_path)
    X_train, y_train, X_test = data_processor.prepareData(drop_columns)

    rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)

    rf_regressor.fit(X_train, y_train)

    y_train_predicitions = rf_regressor.predict(X_train)
    y_test_predicitions = rf_regressor.predict(X_test)

    data_processor.evaluate(y_train, y_train_predicitions)
    data_processor.evaluate(y_test, y_test_predicitions)

    imp_features = rf_regressor.feature_importances_
    feature_names = data_processor.getColumnNames()

    plt.barh(feature_names, imp_features)
    plt.xlabel('Feature importance score')
    plt.ylabel('Features')
    plt.title('Important features of the dataset in a random forest model')
    plt.show()




if __name__ == '__main__':
    main()


