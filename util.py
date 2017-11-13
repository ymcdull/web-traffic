import pandas as pd
import re
import numpy as np
import math

def get_raw_data():
    train = pd.read_csv("train_1.csv")
    test = pd.read_csv("key_1.csv")
    return train, test



def breakdown_topic(str):
    m = re.search('(.*)\_(.*).wikipedia.org\_(.*)\_(.*)', str)
    if m is not None:
        return m.group(1), m.group(2), m.group(3), m.group(4)
    else:
        return "", "", "", ""


def smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 200.0
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0
    return np.nanmean(diff)

def smape_all(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 200.0
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0
    return diff

def get_features(df):
    df['date'] = df['date'].astype('datetime64[ns]')
    df['weekend'] = (df.date.dt.dayofweek // 5).astype(float)
    #df['shortweek'] = ((df.date.dt.dayofweek) // 4 == 1).astype(float)
    return df


def transform_data(train, test, periods=-49):
    train_flattened = pd.melt(train[list(train.columns[periods:])+['Page']], id_vars='Page', var_name='date', value_name='Visits')
    train_flattened = get_features(train_flattened)
    test['date'] = test.Page.apply(lambda a: a[-10:])
    test['Page'] = test.Page.apply(lambda a: a[:-11])
    test = get_features(test)
    return train_flattened, test


