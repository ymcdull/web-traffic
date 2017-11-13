import pickle
import pandas as pd
import numpy as np
import sys

filename = sys.argv[1]

preds = pd.read_csv(filename)

def smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred))
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0
    return 200 * np.mean(diff)

def evaluate(preds):
    tt = pd.read_csv("true_labels.csv")
    print smape(tt.true_labels, preds)

evaluate(preds.Visits)


## save to file
#kk[['Id', 'Visits']].to_csv("sub_lstm_scaler2.csv", index = False)

