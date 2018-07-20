import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import math
import matplotlib.pyplot as plt
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search
from matplotlib.pylab import rcParams


def modelfit(alg, dtrain, dtest, predictors, target, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):

    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            metrics='rmse', early_stopping_rounds=early_stopping_rounds, verbose_eval=True)
        alg.set_params(n_estimators=cvresult.shape[0])

    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain[target],eval_metric='auc')

    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtest_predictions = alg.predict(dtest[predictors])

    #Print model report:
    print("\nModel Report")
    print("RSEM Score (Train): %f" % metrics.mean_squared_error(dtrain[target].values, dtrain_predictions))
    print("RSEM Score (Test): %f" % metrics.mean_squared_error(dtest[target], dtest_predictions))

    feat_imp = pd.Series(
        alg.get_booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
    plt.show()

    return alg, dtest_predictions, feat_imp

data = pd.read_csv('../data/raizen.csv', sep=';')
fac, _ = data.Estagio.factorize()
data.Estagio = fac

fac, _ = data.Ciclo.factorize()
data.Ciclo = fac

fac, _ = data.Variedade.factorize()
data.Variedade = fac

data_y = data.pop('Perc_Falha')
data_x = data

X_train, X_test, Y_train, Y_test = train_test_split(
    data_x, data_y, test_size=0.30, random_state=42)

predictors = X_train.columns
target = 'Perc_Falha'
xgb1 = XGBClassifier(
    learning_rate =0.1,
    n_estimators=1000,
    max_depth=5,
    min_child_weight=1,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    objective= 'binary:logistic',
    nthread=4,
    scale_pos_weight=1,
    seed=27)
train = pd.concat([X_train, Y_train], axis=1)
# train = train[:200]
test = pd.concat([X_test, Y_test], axis=1)
alg, pred, feat_imp= modelfit(
    alg=xgb1, dtrain=train, dtest=test, predictors=predictors, target=target)

