import pickle
import pandas as pd
import numpy as np
import sys

filename = sys.argv[1]
with open(filename, 'r') as f:
    res = pickle.load(f)

df = pd.DataFrame(res)

train = pd.read_csv("train_1.csv", usecols=["Page"])
df["Page"] = train["Page"]

kk = pd.read_csv("key_1.csv")
kk.Page = kk.Page.apply(lambda x: x[:-11])
unique_pages = kk.Page.unique()

new_df = pd.merge(pd.DataFrame(pd.Series(unique_pages), columns=["Page"]), df, how='left')

kk['Visits'] = new_df.iloc[:, 1:].values.reshape(-1, 1)


def smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred))
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0
    return 200 * np.mean(diff)

def evaluate(preds):
    tt = pd.read_csv("true_labels.csv")
    print smape(tt.true_labels, preds)

evaluate(np.expm1(kk.Visits))


## save to file
#kk[['Id', 'Visits']].to_csv("sub_lstm_scaler2.csv", index = False)

