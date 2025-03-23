import numpy as np
import pandas as pd
from LinearRegression import DataProcessing
from tensorflow import keras
import tensorflow as tf

def main():

    train_path = '../clean_data/train_FD001.csv'
    test_path = '../clean_data/test_FD001.csv'

    y_test = pd.read_csv('../CMAPSSData/RUL_FD001.txt', names=['RUL'])

    drop_columns = ['settings 1', 'settings 2', 'settings 3', 'sensor 1'
        , 'sensor 5', 'sensor 6', 'sensor 10', 'sensor 16'
        , 'sensor 18', 'sensor 19']

    data_processor = DataProcessing(train_path, test_path)
    X_train, y_train, X_test = data_processor.prepareData(drop_columns)



    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(1)
    ])

    model.compile(optimizer= tf.keras.optimizers.Adam(learning_rate=0.01), loss='mae')

    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=32,
                        verbose=1)

    y_train_predicitions = model.predict(X_train)
    y_test_predicitions = model.predict(X_test)

    print('Training Data: ')
    data_processor.evaluate(y_train, y_train_predicitions)
    print('-------------------------\nTest Data: ')
    data_processor.evaluate(y_test, y_test_predicitions)


if __name__ == '__main__':
    main()
