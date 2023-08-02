
import pandas as pd
import numpy as np
from math import sin, cos, pi
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import joblib

def preprocess_data(df):
    df = df.copy()
    df['SETTLEMENT_DATE'] = pd.to_datetime(df['SETTLEMENT_DATE'])
    df = df.set_index('SETTLEMENT_DATE')
    df.index = df.index.tz_convert(None)
    return df

def create_features(df):
    df = df.copy()
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    df['dayofmonth'] = df.index.day
    df['weekofyear'] = df.index.isocalendar().week
    return df

def encode_cyclic_feature(df, col, max_val):
    df = df.copy()
    df[col + '_sin'] = np.sin(2 * pi * df[col]/max_val)
    df[col + '_cos'] = np.cos(2 * pi * df[col]/max_val)
    df = df.drop(columns=[col])
    return df

# Load the data
df_2020 = pd.read_csv('demanddata_2020.csv')
df_2021 = pd.read_csv('demanddata_2021.csv')
df_2022 = pd.read_csv('demanddata_2022.csv')
df_2023 = pd.read_csv('demanddata.csv')

# Preprocess the data
df_2020 = preprocess_data(df_2020)
df_2021 = preprocess_data(df_2021)
df_2022 = preprocess_data(df_2022)
df_2023 = preprocess_data(df_2023)

# Add a column for the hour of the day
df_2020['hour'] = (df_2020['SETTLEMENT_PERIOD'] / 2).apply(np.floor)
df_2021['hour'] = (df_2021['SETTLEMENT_PERIOD'] / 2).apply(np.floor)
df_2022['hour'] = (df_2022['SETTLEMENT_PERIOD'] / 2).apply(np.floor)
df_2023['hour'] = (df_2023['SETTLEMENT_PERIOD'] / 2).apply(np.floor)

# Concatenate all the dataframes
df = pd.concat([df_2020, df_2021, df_2022, df_2023])

# Create time-based features
df = create_features(df)

# Encode the 'month' and 'hour' features
df = encode_cyclic_feature(df, 'month', 12)
df = encode_cyclic_feature(df, 'hour', 24)

# Define the features and the target
FEATURES = ['dayofyear', 'hour_sin', 'hour_cos', 'dayofweek', 'quarter', 'month_sin', 'month_cos', 'year']
TARGET = 'ND'

# Split the data into a training set and a test set
train_size = int(len(df) * 0.8)
train, test = df.iloc[:train_size], df.iloc[train_size:]

# Define the training and test sets
X_train = train[FEATURES]
y_train = train[TARGET]

X_test = test[FEATURES]
y_test = test[TARGET]

# Load the model
reg = joblib.load('xgboost_model.pkl')

# Generate predictions on the test set
y_pred = reg.predict(X_test)

# Calculate the root mean square error (RMSE) on the test set
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'RMSE on the test set: {rmse:.2f}')
