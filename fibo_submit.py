## Download Code
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

train = pd.read_csv("train_2.csv")
# train = train.fillna(0.)

Windows = [6, 9, 15, 24 , 39 , 63 , 102 , 165 , 267 , 432]

for W in Windows: 
    train[str(W)]=train.iloc[:,-W:].median(axis=1)

train['Visits']=train.iloc[:,-len(Windows):].median(axis=1)

train.Visits[train.Visits.isnull()] = 0.0
train.Visits[train.Visits < 1] = 0.0

test = pd.read_csv("key_2.csv")
test['date'] = test.Page.apply(lambda x: x[-10:])
test['date'] = test['date'].astype('datetime64[ns]')
test['Page'] = test.Page.apply(lambda x: x[:-11])

test = test.merge(train[['Page','Visits']], on='Page', how='left')

'''
def smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred))
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0
    return 200 * np.mean(diff)

true_labels = pd.read_csv("true_labels.csv")

def evaluate(preds):
    print smape(true_labels.true_labels, preds)

evaluate(test.Visits)
'''

test[['Id', 'Visits']].to_csv('fibo_submit.csv', index = False)
