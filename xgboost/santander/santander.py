import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn import model_selection

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

y = train['TARGET']
X = train.drop(['ID','TARGET'], axis =1)

parameters= {'max_depth': 6,
             'learning_rate': 0.05,
             'n_estimators': 20,
             'objective': 'binary:logitraw',
             'booster': 'gbtree',
             'n_jobs': 4,
             'min_child_weight': 10,
             'max_delta_step': 0,
             'subsample': 0.7,
             'colsample_bytree': 0.5,
             'reg_lambda': 0.9,
             'scale_pos_weight': 23,
             'random_state': 123,
             'base_score': 0.5
             }

clf = xgb.XGBClassifier(**parameters)

clf.fit(X, y)

print(clf.feature_importances_)