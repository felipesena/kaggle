import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Imputer
from xgboost import XGBRegressor


def set_encode(data):
    le = LabelEncoder()
    for column in data.select_dtypes(include=['object']):
        column_value = data[column].fillna('')
        le.fit(column_value)
        data[column] = le.transform(column_value)
        data[column][le.inverse_transform(data[column]) == ''] = np.nan

    return data


warnings.filterwarnings(action='ignore', category=DeprecationWarning)
pd.options.mode.chained_assignment = None

# Read the data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# pull data into target (y) and predictors (X)
train_y = train.SalePrice

# Create training predictors data
train_X = train.drop(columns=['SalePrice', 'Id', 'LotFrontage'], axis=1)
test_X = test.drop(['Id', 'LotFrontage'], axis=1)

train_X = set_encode(train_X)
test_X = set_encode(test_X)

# Preprocessing
imputed_X_train_plus = train_X.copy()
imputed_X_test_plus = test_X.copy()

cols_with_missing = (col for col in train_X.columns
                     if train_X[col].isnull().any())


for col in cols_with_missing:
    imputed_X_train_plus[col + '_was_missing'] = imputed_X_train_plus[col].isnull()
    imputed_X_test_plus[col + '_was_missing'] = imputed_X_test_plus[col].isnull()

# Imputation
my_imputer = Imputer()
imputed_X_train_plus = my_imputer.fit_transform(imputed_X_train_plus)
imputed_X_test_plus = my_imputer.transform(imputed_X_test_plus)

# Fit the model
my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)

my_model.fit(imputed_X_train_plus, train_y, early_stopping_rounds=5,
             eval_set=[(imputed_X_train_plus, train_y)], verbose=False)

# Use the model to make predictions
predicted_prices = my_model.predict(imputed_X_test_plus)

my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_prices})

# you could use any filename. We choose submission here
my_submission.to_csv('submission.csv', index=False)

