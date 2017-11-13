#!/usr/bin/env python
#coding=utf-8

import pandas as pd
import numpy as np
import sys
sys.path.append('..')
from logzero import logger

path_data = ""
def load_nl_data(file):
    df = pd.read_csv(path_data + "%s.csv" % file)
    logger.debug('%s: %s' % (file, df.shape))
    return df

def load_train_ori():
    df_train_ori = pd.read_csv(path_data + "train_1.csv")
    logger.debug('df_train_ori: %s', df_train_ori.shape)
    return df_train_ori

def load_cv_valid():
    cv_valid = pd.read_csv(path_data + "cv_valid_2m.csv")
    cv_valid['dayofweek'] = pd.to_datetime(cv_valid.date).dt.dayofweek
    logger.debug('cv_valid: %s', cv_valid.shape)
    return cv_valid

def load_dataset():
    cv_train = pd.read_csv(path_data + "cv_train_2m.csv")
    cv_valid = pd.read_csv(path_data + "cv_valid_2m.csv")
    train = pd.read_csv(path_data + "train_2m.csv")
    test = pd.read_csv(path_data + "test_2m.csv")
    logger.debug('cv_train: %s', cv_train.shape)
    logger.debug('cv_valid: %s', cv_train.shape)
    logger.debug('train: %s', train.shape)
    logger.debug('test: %s', test.shape)
    return cv_train, cv_valid, train, test


def load_dataset_cv():
    cv_train = pd.read_csv(path_data + "cv_train_2m.csv")
    cv_valid = pd.read_csv(path_data + "cv_valid_2m.csv")
    cv_train = cv_train[~pd.isnull(cv_train.traffic)]
    logger.debug('cv_train: %s', cv_train.shape)
    logger.debug('cv_valid: %s', cv_train.shape)
    return cv_train, cv_valid

def load_dataset_pred():
    train = pd.read_csv(path_data + "train_2m.csv")
    test = pd.read_csv(path_data + "test_2m.csv")
    train = train[~pd.isnull(train.traffic)]
    logger.debug('train: %s', train.shape)
    logger.debug('test: %s', test.shape)
    return train, test


def load_dataset_nl():
    cv_train = pd.read_csv(path_data + "nl_cv_train2.csv")
    cv_valid = pd.read_csv(path_data + "nl_cv_valid1.csv")
    train = pd.read_csv(path_data + "nl_train2.csv")
    test = pd.read_csv(path_data + "nl_test1.csv")
    logger.debug('cv_train: %s, mean: %s' % (cv_train.shape, cv_train.Visits.mean()))
    logger.debug('cv_valid: %s, mean: %s' % (cv_valid.shape, cv_valid.Visits.mean()))
    logger.debug('train: %s, mean: %s' % (train.shape, train.Visits.mean()))
    logger.debug('test: %s, mean: %s' % (test.shape, 'nan'))
    return cv_train, cv_valid, train, test


def load_test():
    test = pd.read_csv(path_data + "test.csv")
    logger.debug('test: %s', test.shape)
    return test

def load_test_page_label():
    test = pd.read_csv(path_data + "page_lab.csv.v3", names=["Page","date","Visits"])
    logger.debug('page_label: %s', test.shape)
    return test

def load_bad_page():
    bad  = pd.read_csv(path_data + "bad_page_cv_2w_miss.csv", sep="%", names=["Page"])
    logger.debug('bad: %s', bad.shape)
    return bad

def load_dataset_all():
    train_all = pd.read_csv(path_data + "train_all.csv")
    logger.debug('train_all: %s', train_all.shape)
    return train_all

def load_key():
    df_key = pd.read_csv(path_data + 'key_1.csv')
    df_key['date'] = pd.to_datetime(df_key.Page.apply(lambda x: x[-10:]))
    df_key['Page'] = df_key.Page.apply(lambda x: x[:-11])
    df_key['dayofweek'] = df_key.date.dt.dayofweek
    logger.debug('df_key: %s', df_key.shape)
    return df_key 

def load_train_page_date_info():
    df = pd.read_csv(path_data + 'train_page_date_info.csv')
    logger.debug('train_page_date_info: %s', df.shape)
    return df

def load_test_page_date_info():
    df = pd.read_csv(path_data + 'test_page_date_info.csv')
    logger.debug('test_page_date_info: %s', df.shape)
    return df

def load_page_info():
    df_page_info = pd.read_csv(path_data + "page_info.csv")
    logger.debug('df_page_info: %s', df_page_info.shape)
    return df_page_info

def load_page_feat():
    df_page_feat = pd.read_csv(path_feature + "page_static_feat.csv")
    logger.debug('page_static_feat: %s', df_page_feat.shape)
    return df_page_feat

def load_sample_stat_feat(feat_file):
    df_feat = pd.read_csv(path_feature + "%s.csv" % feat_file)
    logger.debug('%s: %s' % (feat_file, df_feat.shape))
    return df_feat

def load_feat(feat_file):
    df_feat = pd.read_csv(path_feature + "%s.csv" % feat_file)
    logger.debug('%s: %s' % (feat_file, df_feat.shape))
    return df_feat


if __name__ == '__main__':
    #df_train_ori = load_train_ori()
    #df_train = pd.melt(df_train_ori, id_vars='Page', var_name='date', value_name='traffic')
    #df_train.to_csv(path_data + 'sample_ori.csv', index=False)
    #df_page_index = df_train_ori[['Page']]
    #df_page_index['index'] = df_train_ori.index
    #df_page_index.to_csv(path_data + 'page_index.csv', index=False)
    #cv_train, cv_valid, train, test = load_dataset()
    #data = pd.concat([cv_train,cv_valid,test], axis = 0)
    #data.to_csv(path_data + 'train_all.csv',index=False)
    load_key()

