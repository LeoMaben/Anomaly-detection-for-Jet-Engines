import pandas as pd
import numpy as np

data_path = '../CMAPSSData/'
clean_path = '../clean_data/'

operational_names = ['settings 1', 'settings 2', 'settings 3']
sensor_names = ['sensor {}'.format(i) for i in range(1, 22)]

coloumns = ['unit number', 'time cycles'] + operational_names + sensor_names

# read data
train = pd.read_csv((data_path +'train_FD001.txt'), sep='\s+', header=None, names=coloumns)
test = pd.read_csv((data_path +'test_FD001.txt'), sep='\s+', header=None, names=coloumns)
y_test = pd.read_csv((data_path +'RUL_FD001.txt'), sep='\s+', header=None, names=['RUL'])

print(train.head())
# print(y_test.head())
# print(test.head())

# save data
train.to_csv(clean_path + 'train_FD001.txt', index=False)
test.to_csv(clean_path + 'test_FD001.txt', index=False)
y_test.to_csv(clean_path + 'RUL_FD001.txt', index=False)


