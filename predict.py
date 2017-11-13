import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.callbacks import ModelCheckpoint
import pickle

np.random.seed(7)


from keras import backend as K

def smape(y_true, y_pred):
    diff = K.abs((y_true - y_pred) / K.clip(K.abs(y_true) + K.abs(y_pred),
                                            K.epsilon(),
                                            None))
    return 200. * K.mean(diff, axis=-1)


train = pd.read_csv("train_1.csv")
train = train.fillna(0.0)

# train = train.iloc[:, -55:]
train = train.iloc[:, -120:]

train = train.values.astype('float32')

X_train = train
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)


# create and fit the LSTM network
model = Sequential()
model.add(LSTM(200, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(100, return_sequences = False))
model.add(Dropout(0.2))
model.add(Dense(60))

# load weights
model.load_weights("weights-improvement-16-45.93.hdf5")

# model.compile(loss='mean_absolute_error', optimizer='adam')
model.compile(loss=smape, optimizer='adam')

res = model.predict(X_train)

with open("predict.pkl", 'w') as f:
    pickle.dump(res, f)
