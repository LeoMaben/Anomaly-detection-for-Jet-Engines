import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

train_path = '../clean_data/train_FD001.csv'
test_path ='../clean_data/test_FD001.csv'
train_df = pd.read_csv(train_path)

drop_coloumns = ['settings 1', 'settings 2', 'settings 3', 'sensor 1',  'sensor 5', 'sensor 6', 'sensor 10',
                 'sensor 16', 'sensor 18', 'sensor 19']

train_df = train_df.drop(drop_coloumns, axis=1)
train_df['RUL'] = train_df.groupby('unit number')['time cycles'].transform(max) - train_df['time cycles']
train_df = train_df.drop(['unit number', 'time cycles'], axis=1)

X = train_df.drop('RUL', axis=1)
y = train_df['RUL']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LinearRegression()
model.fit(X_scaled, y)

# predict and evaluate for training set
y_train_predictions = model.predict(X_scaled)

mae = mean_absolute_error(y, y_train_predictions)
rmse = np.sqrt(mean_squared_error(y, y_train_predictions))
r2 = r2_score(y, y_train_predictions)

print(f'Mean Absolute Error is: {mae}\n'
      f'Mean Squared Error is: {rmse}\n'
      f'R2 Score is: {r2}')
