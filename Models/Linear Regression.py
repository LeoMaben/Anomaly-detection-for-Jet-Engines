import pandas as pd
# from clean_data import sensor_names
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

def plot_sensors(sensor_name):
    plt.figure(figsize=(13, 5))
    for i in test_data['unit number'].unique():
        if (i % 10 == 0):  # only plot every 10th unit_nr
            plt.plot('RUL', sensor_name,
                     data=test_data[test_data['unit number'] == i])
    plt.xlim(250, 0)  # reverse the x-axis so RUL counts down to zero
    plt.xticks(np.arange(0, 275, 25))
    plt.ylabel(sensor_name)
    plt.xlabel('Remaining Use fulLife')
    plt.show()


def evaluate_rmse(y_true, y_pred):
    """
    Calculate Root Mean Square Error (RMSE) between true and predicted values.

    Parameters:
    y_true : Actual values
    y_pred : Predicted values
    """

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    variance = r2_score(y_true, y_pred)
    print(f'The RMSE is: {rmse} and the R-squared is {variance}')





input_path = '../clean_data/'

train_data = pd.read_csv(input_path + 'train_FD001.txt')
test_data = pd.read_csv(input_path + 'test_FD001.txt')
y_test = pd.read_csv(input_path + 'RUL_FD001.txt')
index_names = ['unit number', 'time cycles']

#print(test_data[index_names].describe())
#print(test_data[index_names].groupby('unit number').max().describe())
#print(test_data[['settings 1', 'settings 2', 'settings 3']].describe()) # Only Settings 3 is being used
#print(test_data[sensor_names].describe().transpose())

train_data['RUL'] = train_data.groupby('unit number')['time cycles'].transform('max') - train_data['time cycles']
drop_sensors = ['sensor 1', 'sensor 5', 'sensor 6', 'sensor 10', 'sensor 16', 'sensor 18', 'sensor 19']
drop_labels = index_names+['settings 1', 'settings 2', 'settings 3']+drop_sensors
print(train_data)
x_train = train_data.drop(drop_labels, axis=1)
y_train = x_train.pop('RUL')
print(y_train)


x_test = test_data.groupby('unit number').last().reset_index().drop(drop_labels, axis=1)

y_train = y_train.clip(upper=125)

linear_model = LinearRegression()
linear_model.fit(x_train, y_train)

# Predict training data
y_pred = linear_model.predict(x_train)

# Evaluate training data
evaluate_rmse(y_train, y_pred)

# Predict & Evaluate test data
y_pred_test = linear_model.predict(x_test)
evaluate_rmse(y_test, y_pred_test)



