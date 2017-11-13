#!/usr/bin/env python
#coding=utf-8

import pandas as pd
import numpy as np
import sys
sys.path.append('..')
from load import load_train_ori

df_train_ori = load_train_ori()
path_data = ""

page_info = pd.DataFrame({'Page': df_train_ori.Page.unique()})
# split it to add some features
page_info['agent_type'] = page_info.Page.str.rsplit('_').str.get(-1)
page_info['access_type'] = page_info.Page.str.rsplit('_').str.get(-2)
page_info['project'] = page_info.Page.str.rsplit('_').str.get(-3)
# dirty hacking to get it :)
page_info['page_name'] = page_info.apply(
		    lambda r: r['Page'][:-int(len(r['agent_type'])+len(r['access_type'])+len(r['project'])+3)], axis=1)
# add country
page_info['source'] = page_info.project.str.split('.').str.get(0).map(
		{'en':'English', 'ja':'Japanese', 'de':'German', 'fr':'France', 'zh':'Chinese',
			'ru':'Russian', 'es':'Spanish','commons':'wikimedia','www':'mediawiki'})
page_info['url'] = page_info.project + '/wiki/' +  page_info.page_name
page_info['page_type'] = page_info.page_name.apply(lambda x: x.rsplit('.')[-1] if len(x.rsplit('.'))>1 else 'page')
page_info['index'] = df_train_ori.index
print page_info.shape
page_info.to_csv(path_data + "page_info.csv", index=False)

#df_train_flattened = pd.melt(df_train_ori, id_vars='Page', var_name='date', value_name='traffic')
#df_train_flattened.to_csv(path_data + "train.csv")
