import pandas
import pandas as pd
from typing import List
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def prepareTrainData(train_df:pandas.DataFrame, drop_coloumns:List[str], scaler):
    train_df = train_df.drop(drop_coloumns, axis=1)
    train_df['RUL'] = train_df.groupby('unit number')['time cycles'].transform(max) - train_df['time cycles']
    train_df = train_df.drop(['unit number', 'time cycles'], axis=1)

    X = train_df.drop('RUL', axis=1)
    y = train_df['RUL'].clip(upper=125)

    X_scaled = scaler.fit_transform(X)


    return X_scaled, y


def evaluate(y, y_predictions):
    mae = mean_absolute_error(y, y_predictions)
    rmse = np.sqrt(mean_squared_error(y, y_predictions))
    r2 = r2_score(y, y_predictions)


    print(f'Mean Absolute Error is: {mae}\n'
          f'Mean Squared Error is: {rmse}\n'
          f'R2 Score is: {r2}')


def main():
    train_path = '../clean_data/train_FD001.csv'
    test_path ='../clean_data/test_FD001.csv'
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    y_test = pd.read_csv('../CMAPSSData/RUL_FD001.txt', names=['RUL'])


    drop_coloumns = ['settings 1', 'settings 2', 'settings 3', 'sensor 1',  'sensor 5', 'sensor 6', 'sensor 10',
                     'sensor 16', 'sensor 18', 'sensor 19']
    scaler = StandardScaler()

    X_train_scaled, y_train = prepareTrainData(train_df, drop_coloumns, scaler)

    X_test = test_df.groupby('unit number').last().reset_index()
    X_test_scaled, _ = prepareTrainData(X_test, drop_coloumns, scaler)

    poly = PolynomialFeatures(3)
    X_train = poly.fit_transform(X_train_scaled)
    X_test = poly.fit_transform(X_test_scaled)


    model = LinearRegression()
    model.fit(X_train, y_train)

    # predict and evaluate for training set
    y_train_predictions = model.predict(X_train)
    y_test_predicitions = model.predict(X_test)

    print('Training Data: ')
    evaluate(y_train, y_train_predictions)
    print('-------------------------\nTest Data: ')
    evaluate(y_test, y_test_predicitions)





if __name__ == '__main__':
    main()