import os
import sys
import math
import numpy as np
import pandas as pd
from bayes_opt import BayesianOptimization
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, RepeatedStratifiedKFold, cross_val_score
from sklearn.metrics import f1_score, confusion_matrix, matthews_corrcoef, hamming_loss, jaccard_score
from sklearn.preprocessing import LabelEncoder 
from bayes_opt.util import Colours
import matplotlib.pyplot as plt
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events



def get_data():
    """Do all the reading and preprocessing of the data
    This function returns an X
    """
    train = pd.read_csv(os.path.join(os.getcwd(),'train.csv'))
    test = pd.read_csv(os.path.join(os.getcwd(),'test.csv'))

    X = train.loc[:, 'Elevation':'Soil_Type']
    Y = train.loc[:, 'Cover_Type'::]
    X_test = test.loc[:,'Elevation':]

    le = LabelEncoder()
    X['Wilderness_Area'] = le.fit_transform(X['Wilderness_Area'])
    X_test['Wilderness_Area'] = le.fit_transform(X_test['Wilderness_Area'])
    Y['Cover_Type'] = le.fit_transform(Y['Cover_Type'])

#     X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.33, random_state=5342792)

    y_train = np.ravel(Y)
#     y_val = np.ravel(y_val)

    X_tr_ar = np.array(X)
#     X_va_ar = np.array(X_val)
    X_te_ar = np.array(X_test)

    X_tr_norm = np.empty((X_tr_ar.shape[0],X_tr_ar.shape[1]))
#     X_te_norm = np.zeros((X_va_ar.shape[0],X_va_ar.shape[1]))
    X_test_norm = np.empty((X_te_ar.shape[0],X_te_ar.shape[1]))

    for i in range(X_tr_ar.shape[1]):
        X_tr_norm[:,i] = (X_tr_ar[:,i] - np.mean(X_tr_ar[:,i],axis=0)) / np.sqrt(np.var(X_tr_ar[:,i],axis=0))
#         X_te_norm[:,i] = (X_va_ar[:,i] - np.mean(X_va_ar[:,i],axis=0)) / np.sqrt(np.var(X_va_ar[:,i],axis=0))
        X_test_norm[:,i] = (X_te_ar[:,i] - np.mean(X_te_ar[:,i],axis=0)) / np.sqrt(np.var(X_te_ar[:,i],axis=0))
    
    return X_tr_norm, y_train, X_test_norm
    
    

def rfc_cv(n_estimators, max_depth, min_samples_split, min_samples_leaf, data, targets):
    """Random Forest cross validation.
    This function will instantiate a random forest classifier with parameters
    n_estimators, min_samples_split, and max_features. Combined with data and
    targets this will in turn be used to perform cross validation. The result
    of cross validation is returned.
    Our goal is to find combinations of n_estimators, min_samples_split, and
    max_features that minimzes the log loss.
    """
    estimator = RandomForestClassifier(
        n_estimators=n_estimators, 
        max_depth=max_depth, 
        min_samples_split=min_samples_split, 
        bootstrap=False,
        min_samples_leaf=min_samples_leaf,
        random_state=np.random.randint(1,654321)
    )
    cval = cross_val_score(estimator, data, targets,
                           scoring='neg_log_loss', cv=2)
    return cval.mean()



def optimize_rfc(data, targets):
    """Apply Bayesian Optimization to Random Forest parameters."""
    def rfc_crossval(n_estimators, max_depth, min_samples_split, min_samples_leaf): #, max_features):
        """Wrapper of RandomForest cross validation.
        Notice how we ensure n_estimators and min_samples_split are casted
        to integer before we pass them along. Moreover, to avoid max_features
        taking values outside the (0, 1) range, we also ensure it is capped
        accordingly.
        """
        return rfc_cv(
            n_estimators=int(n_estimators),
            max_depth=int(max_depth),
            min_samples_split=int(min_samples_split),
            min_samples_leaf=int(min_samples_leaf),
            data=data,
            targets=targets,
        )

    optimizer = BayesianOptimization(
        f=rfc_crossval,
        pbounds={
            "n_estimators": (750, 1250),
            "max_depth": (50, 150),
            "min_samples_split": (2, 25),
            "min_samples_leaf": (2, 15),
#             "max_features": (0.1, 0.999),
        },
        random_state=1234,
        verbose=2
    )

    # logger = JSONLogger(path="./logs.json")
    # optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
    
    optimizer.maximize(
        init_points=20,
        n_iter=80,
        acq='ucb',
        kappa=5,
        # acq="ei", 
        # xi=1.e-04,
        # acq="poi", 
        # xi=1.e-04
    )

    print("Final result:\n", optimizer.max)


if __name__ == "__main__":
    train_x, train_y, test_x = get_data()

#     print(Colours.yellow("--- Optimizing SVM ---"))
#     optimize_svc(data, targets)

    print(Colours.green("--- Optimizing Random Forest ---"))
    optimize_rfc(train_x, train_y)
    
    

# def crossvalidate(xTr, yTr, ktype, Cs, paras):
#     bestC, bestP, lowest_error = 0, 0, 0
#     errors = np.zeros((len(paras),len(Cs)))
    
#     def _scoring_metric(expC=1.0,expGamma=0.475):
#         skf = RepeatedStratifiedKFold(n_splits=10, n_repeats=10)
#         scores = np.empty((skf.get_n_splits(),))
#         for idx, (train_index, test_index) in enumerate(skf.split(xTr.T, yTr)):
#             # print(f'idx: {idx}')
#             # print(f'TRAIN: {train_index}, TEST: {test_index}')
#             X_train, X_test = xTr.T[train_index].T, xTr.T[test_index].T
#             y_train, y_test = yTr[train_index], yTr[test_index]
#             C = 10. ** expC          # we transform from log scale to 
#             gamma = 10. ** expGamma  # improve optimizer performance
#             svmclassify = train_rfc(X_train, y_train, C, ktype, gamma)
#             y_preds = svmclassify(X_test)
#             # scores[idx] = 1. - np.mean(y_preds != y_test) # 1 - training_error = accuracy
#             # scores[idx] = f1_score(y_test, y_preds) # f1 score
#             # try:
#             #     # needs to catch divide by zero exceptions
#             #     scores[idx] = matthews_corrcoef(y_test, y_preds)
#             # except RuntimeWarning as _:
#             #     # use f1 if mcc is undefined
#             #     scores[idx] = f1_score(y_test, y_preds) 
#             # scores[idx] = -hamming_loss(y_test, y_preds) # hamming loss
#             scores[idx] = jaccard_score(y_test, y_preds) # jaccard similarity
#             # tn, fp, fn, tp = confusion_matrix(y_test, y_preds).ravel()
#             # scores[idx] = (fp / (fp + tp)) # false discovery rate
#         return np.mean(scores)

#     # Bounded region of parameter space
#     pbounds = {'expC': (0, 2), 'expGamma': (0, 1)}

#     # Create the optimizer object
#     optimizer = BayesianOptimization(
#         f=_scoring_metric,
#         pbounds=pbounds,
#         verbose=1,
#         random_state=np.random.randint(1,654321)
#     )

#     # Probe points you already know are near the optimum
# #     good_Cs = [np.log10(6.053), np.log10(16.99), np.log10(52.62), np.log10(31.35), np.log10(17.38),
# #                np.log10(88.06), np.log10(60.53), np.log10(92.69), np.log10(29.67), np.log10(25.62),
# #                np.log10(72.47), 1.122, 0.7014, 0.708, 0.9839, 0.6773, 1.167, 1.766, 1.668, 1.039, 
# #                0.6037, 1.002, 0.9607, 1.141, 1.349]
# #     good_gammas = [np.log10(3.00), np.log10(3.00), np.log10(2.90), np.log10(2.90),np.log10(2.80), 
# #                    np.log10(2.80), np.log10(3.015), np.log10(3.26), np.log10(2.727),np.log10(2.951), 
# #                    np.log10(2.949), 0.4329, 0.4455, 0.4322, 0.4679, 0.4422, 0.4446, 0.4384, 0.4223, 
# #                    0.4761, 0.4473, 0.4324, 0.5026, 0.4748, 0.4365]
    
#     for regularization_param, kernel_param in zip(good_Cs, good_gammas):
#         optimizer.probe(
#             params={'expC': regularization_param, 'expGamma': kernel_param},
#             lazy=True,
#         )
    
#     # Perform the rest of the search
#     optimizer.maximize(
#         init_points=100,
#         n_iter=300,
#         acq='ucb',
#         kappa=1.0,
#         # acq="ei", 
#         # xi=1.e-04,
#         # acq="poi", 
#         # xi=1.e-04
#     )

#     # plot parameter surface (only handles 1D functions atm)
#     # plot_bo(_scoring_metric, optimizer)

#     lowest_error, bestC, bestP = optimizer.max['target'], 10.**optimizer.max['params']['expC'], 10.**optimizer.max['params']['expGamma']

#     return bestC, bestP, lowest_error, errors


####################################
# Utility function for plotting GP #
####################################
def plot_bo(f, bo):
    print(bo.space.params)
    sys.exit(1)
    C = np.logspace(0, 2, 10000)
    gamma = np.logspace(0, 1, 10000)
    X, Y = np.meshgrid(C, gamma)
    mean, sigma = bo._gp.predict(x.reshape(-1, 1), return_std=True)
    
    plt.figure(figsize=(16, 9))
    # plt.plot(x, f(x))
    plt.plot(x, mean)
    plt.fill_between(x, mean + sigma, mean - sigma, alpha=0.1)
    plt.scatter(bo.space.params.flatten(), bo.space.target, c="red", s=50, zorder=10)
    plt.show()
    
