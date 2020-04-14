# 441975, l.teixeira@wustl.edu, Teixeira, Lucas
# 443896, rickynoll@wustl.edu, Noll, Ricky
# 463379, dariakowsari@wustl.edu, Kowsari, Daria

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder 
from sklearn.metrics import accuracy_score

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

X = train.loc[:, 'Elevation':'Soil_Type']
Y = train.loc[:, 'Cover_Type'::]
X_test = test.loc[:,'Elevation':]

le = LabelEncoder()
X['Wilderness_Area'] = le.fit_transform(X['Wilderness_Area'])
X_test['Wilderness_Area'] = le.fit_transform(X_test['Wilderness_Area'])
Y['Cover_Type'] = le.fit_transform(Y['Cover_Type'])

X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.33, random_state=5342792)

y_train = np.ravel(y_train)
y_val = np.ravel(y_val)

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

rf = RandomForestClassifier(n_estimators=1000, max_depth = 100, min_samples_split = 8, bootstrap=False,\
                            min_samples_leaf = 5)

rf.fit(X_tr_norm, y_train)

predictions = rf.predict(X_te_norm)

# Calculate the errors
Acc = accuracy_score(predictions,y_val)
print('Cross Validation ACC:', Acc)

y_test_pred = rf.predict(X_test_norm)
y_test_pred = le.inverse_transform(y_test_pred)

my_submission = pd.DataFrame({'ID': test.ID, 'Cover_Type': y_test_pred})
my_submission.to_csv('result_norm_RanFor.csv', index=False)
