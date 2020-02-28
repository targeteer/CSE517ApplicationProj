# 441975, l.teixeira@wustl.edu, Teixeira, Lucas
# 443896, rickynoll@wustl.edu, Noll, Ricky
# 463379, dariakowsari@wustl.edu, Kowsari, Daria


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split

from grdscent import grdescent
from linearmodel import linearmodel
from ridge import ridge
from hinge import hinge
from logistic import logistic


train = pd.read_csv('train.csv', index_col=0)
X = train.loc[:, 'Elevation':'Soil_Type']
y = train.loc[:, 'Horizontal_Distance_To_Fire_Points'::]


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state=5342792)
X_tr_ar = np.array(X_train)
X_va_ar = np.array(X_val)

X_tr_norm = np.zeros((X_tr_ar.shape[0],X_tr_ar.shape[1]))
X_te_norm = np.zeros((X_va_ar.shape[0],X_va_ar.shape[1]))

for i in range(X_tr_ar.shape[1]):
    X_tr_norm[:,i] = (X_tr_ar[:,i] - np.mean(X_tr_ar[:,i],axis=0)) / np.sqrt(np.var(X_tr_ar[:,i],axis=0))
    X_te_norm[:,i] = (X_va_ar[:,i] - np.mean(X_va_ar[:,i],axis=0)) / np.sqrt(np.var(X_va_ar[:,i],axis=0))

print("X_train: ", X_train.shape)
print("X_val: ",X_val.shape)
print("y_train: ",y_train.shape)
print("y_val: ",y_val.shape)

y_train = np.ravel(y_train)
y_val = np.ravel(y_val)

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

#model = linear_model.LinearRegression()
model = linear_model.LogisticRegression(max_iter=10000000,tol=1e-07,penalty='l2')
#model = linear_model.SGDClassifier(loss='log',penalty='l2',tol=1e-12, max_iter=100000000, alpha=0.00000001)
#rig = linear_model.LogisticRegression()
#clf = linear_model.Lasso(alpha=0.1)

model.fit(X_tr_norm, y_train)
#rig.fit(X_train, y_train)
#clf.fit(X_train, y_train)

y_pred = model.predict(X_te_norm)
y_pred = np.ravel(y_pred)
#y_pred_rig = rig.predict(X_val)
#y_pred_sgd = clf.predict(X_val)

"""
# ImpProj1 Functions
w_trained = np.zeros((X_train.shape[1],1))
f = lambda w : hinge(w,X_train.T,y_train.T,0.0000000001)
w_trained = grdescent(f,np.zeros((X_train.shape[1],1)),10,1000000000, 1e-10)

y_pred = linearmodel(w_trained, X_val.T)
"""

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
 