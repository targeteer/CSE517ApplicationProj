# 441975, l.teixeira@wustl.edu, Teixeira, Lucas
# 443896, rickynoll@wustl.edu, Noll, Ricky
# 463379, dariakowsari@wustl.edu, Kowsari, Daria
# 437008, donggyukim@wustl.edu, Kim, Donggyu

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

X = train.loc[:, 'Elevation':'Soil_Type']
y = train.loc[:, 'Horizontal_Distance_To_Fire_Points'::]
X_test = test.loc[:,'Elevation':]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state=5342792)
X_tr_ar = np.array(X_train)
X_va_ar = np.array(X_val)
X_te_ar = np.array(X_test)

X_tr_norm = np.zeros((X_tr_ar.shape[0],X_tr_ar.shape[1]))
X_te_norm = np.zeros((X_va_ar.shape[0],X_va_ar.shape[1]))
X_test_norm = np.zeros((X_te_ar.shape[0],X_te_ar.shape[1]))

for i in range(X_tr_ar.shape[1]):
    X_tr_norm[:,i] = (X_tr_ar[:,i] - np.mean(X_tr_ar[:,i],axis=0)) / np.sqrt(np.var(X_tr_ar[:,i],axis=0))
    X_te_norm[:,i] = (X_va_ar[:,i] - np.mean(X_va_ar[:,i],axis=0)) / np.sqrt(np.var(X_va_ar[:,i],axis=0))
    X_test_norm[:,i] = (X_te_ar[:,i] - np.mean(X_te_ar[:,i],axis=0)) / np.sqrt(np.var(X_te_ar[:,i],axis=0))

print("X_train: ", X_train.shape)
print("X_val: ",X_val.shape)
print("y_train: ",y_train.shape)
print("y_val: ",y_val.shape)

y_train = np.ravel(y_train)
y_val = np.ravel(y_val)

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

model = linear_model.LogisticRegression(max_iter=100000,tol=1e-09,penalty='l2')

model.fit(X_tr_norm, y_train)

y_pred = model.predict(X_te_norm)
y_pred = np.ravel(y_pred)

y_test_pred = model.predict(X_test_norm)

my_submission = pd.DataFrame({'ID': test.ID, 'Horizontal_Distance_To_Fire_Points': y_test_pred})
my_submission.to_csv('result_norm_LogReg.csv', index=False)

mse = mean_squared_error(y_val, y_pred)
mae = mean_absolute_error(y_val, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_val, y_pred)

print("--- MSE ---")
print(mse)
print("--- RMSE ---")
print(rmse)
print("--- MAE ---")
print(mae)
print("--- r2 ---")
print(r2)
 