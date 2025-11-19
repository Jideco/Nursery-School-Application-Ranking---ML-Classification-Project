#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import sklearn
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction import DictVectorizer
import xgboost as xgb

print(f'pandas=={pd.__version__}')
print(f'numpy=={np.__version__}')
print(f'sklearn=={sklearn.__version__}')

# data preparation

def load_data():

    df = pd.read_csv('data/nursery.csv')

    encoder = LabelEncoder()
    encoder.fit(df['class'])
    df['class'] = encoder.transform(df['class'])

    return df


def train_model(df):
    categorical = ['parents', 'has_nurs', 'form', 'children','housing', 'finance', 'social','health']

    train_dicts = df[categorical].to_dict(orient='records')
    y_train = df['class'].values

    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(train_dicts)

    features = list(dv.get_feature_names_out())
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=features)

    xgb_params = {
        'eta': 0.1,
        'max_depth': 6,
        'min_child_weight': 1,

        'objective': 'multi:softmax',
        'num_class' : 5,

        'eval_metric': ['merror', 'mlogloss'],

        'nthread': 8,
        'seed': 1,
        'verbosity': 1
        }

    model = xgb.train(xgb_params, dtrain, num_boost_round=175)

    return dv, model


def save_model(filename, dv, model):
    with open(filename, 'wb') as f_out:
        pickle.dump((dv, model), f_out)

    print('model saved to %s' % (filename))


df = load_data()
dv, model = train_model(df)
save_model('model.bin', dv, model)