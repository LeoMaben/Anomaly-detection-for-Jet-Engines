import pandas as pd
from typing import List
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class DataProcessing:
    def __init__(self, train_path:str, test_path: str):
        self.names = []
        self.train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        test_df = test_df.groupby('unit number').last().reset_index()
        self.test_df = test_df

    def __preprocessData(self, df, drop_columns) -> pd.DataFrame :

        df = df.drop(drop_columns, axis=1)
        df['RUL'] = df.groupby('unit number')['time cycles'].transform(max) - df['time cycles']
        df = df.drop(['unit number', 'time cycles'], axis=1)

        return df

    def prepareData(self, drop_columns: List[str]):

        train_df = self.__preprocessData(self.train_df, drop_columns)
        test_df = self.__preprocessData(self.test_df, drop_columns)

        self.names = self.__setColumnNames(train_df.drop('RUL', axis=1))

        scaler = StandardScaler()

        X_train = train_df.drop('RUL', axis=1)
        y_train = train_df['RUL'].clip(upper=125)

        X_test = test_df.drop('RUL', axis=1)

        X_scaled_train = scaler.fit_transform(X_train)
        X_scaled_test = scaler.fit_transform(X_test)

        return X_scaled_train, y_train, X_scaled_test

    def evaluate(self, y, y_predictions):
        mae = mean_absolute_error(y, y_predictions)
        rmse = np.sqrt(mean_squared_error(y, y_predictions))
        r2 = r2_score(y, y_predictions)

        print(f'Mean Absolute Error is: {mae}\n'
              f'Mean Squared Error is: {rmse}\n'
              f'R2 Score is: {r2}')

    def __setColumnNames(self, df):
        return df.columns

    def getColumnNames(self):
        return self.names

def main():
    train_path = '../clean_data/train_FD001.csv'
    test_path ='../clean_data/test_FD001.csv'

    y_test = pd.read_csv('../CMAPSSData/RUL_FD001.txt', names=['RUL'])

    drop_columns = ['settings 1', 'settings 2', 'settings 3', 'sensor 1'
                    , 'sensor 5', 'sensor 6', 'sensor 10', 'sensor 16'
                    , 'sensor 18', 'sensor 19']

    data_processor = DataProcessing(train_path, test_path)
    X_train, y_train, X_test = data_processor.prepareData(drop_columns)


    # poly = PolynomialFeatures(3)
    # X_train = poly.fit_transform(X_train)
    # X_test = poly.fit_transform(X_test)


    model = LinearRegression()
    model.fit(X_train, y_train)

    # predict and evaluate for training set
    y_train_predictions = model.predict(X_train)
    y_test_predicitions = model.predict(X_test)

    print('Training Data: ')
    data_processor.evaluate(y_train, y_train_predictions)
    print('-------------------------\nTest Data: ')
    data_processor.evaluate(y_test, y_test_predicitions)



if __name__ == '__main__':
    main()