from dbm import error
from xml.etree.ElementInclude import include

import numpy as np
import pandas as pd
from fontTools.subset import subset
from sqlalchemy.dialects.mssql.information_schema import columns
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error

da = pd.read_csv('../Database_Regressions/data_akbilgic.csv')
# print(da.head().to_string())
# print(da.columns)
# print(da.shape)
# print(da.info())
# print(da.describe().to_string())
# print(da.isna().sum())


# print(da.head().to_string())
# print(x.isna().sum())

''''[Unnamed: 0', 'TL BASED', 'USD BASED', 'imkb_x', 'Unnamed: 4',
       'Unnamed: 5', 'Unnamed: 6', 'Unnamed: 7', 'Unnamed: 8', 'Unnamed: 9']'''

da = da.iloc[:, 1:]
numeric_cols = da.columns
da[numeric_cols] = da[numeric_cols].apply(pd.to_numeric, errors='coerce')

target = 'imkb_x'
da = da.dropna(subset=[target])

da = da.fillna(da.mean())

for col in da.select_dtypes(include=np.number).columns:
    da[col] = np.sqrt(da[col].clip(lower=0))


# print(da.isna().sum())

# #
# for col in da.select_dtypes(include='number').columns:
#     Q1 = da[col].quantile(0.25)
#     Q3 = da[col].quantile(0.75)
#     IQR = Q3 - Q1
#
#     lower_bound = Q1 - 1.5 * IQR
#     upper_bound = Q3 + 1.5 * IQR
#
#     outliers = da[(da[col] < lower_bound) | (da[col] < upper_bound)]
#
#     print(f'column:{col}')
#     print(f'lower:{lower_bound}')
#     print(f'upper:{upper_bound}')
#     print(f'count_outlires:{len(outliers)}')
#
#     if len(outliers)==0:
#         print('No OutLier')
#
#     print('-'*40)
#
# print(100*'*')
def remove_outlier(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return (df[(df[column] >= lower_bound) & (df[column] <= upper_bound)])


for column in ['TL BASED', 'USD BASED', 'imkb_x', 'Unnamed: 4',
               'Unnamed: 5', 'Unnamed: 6', 'Unnamed: 7', 'Unnamed: 8', 'Unnamed: 9']:
    da = remove_outlier(da, column)

x = da.drop(['TL BASED'], axis=1)
y = da['TL BASED']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
# print(x_train.shape)
# print(x_test.shape)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(x_train)
X_test_scaled = scaler.transform(x_test)

lr_model = LinearRegression()
lr_model.fit(x_train, y_train)

y_pred = lr_model.predict(x_test)
# print(y_pred)
# print(len(y_pred))
# print(len(y_test))

MSE = mean_squared_error(y_test, y_pred)
RMSE = np.sqrt(MSE)
print('MSE:', MSE)
print('RMSE:', RMSE)

final_model = LinearRegression()
final_model.fit(x.values, y)
y_hat = final_model.predict(x.values)
print(y_hat)
