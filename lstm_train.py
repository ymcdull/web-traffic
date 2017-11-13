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

train = pd.read_csv("train_2.csv")

from keras import backend as K

def smape(y_true, y_pred):
    diff = K.abs((y_true - y_pred) / K.clip(K.abs(y_true) + K.abs(y_pred),
                                            K.epsilon(),
                                            None))
    return 200. * K.mean(diff, axis=-1)

train = train.fillna(0.0)
train = train.iloc[:, -210:]
train = train.values.astype('float32')

X_train, Y_train = train[:, -210: -90], train[:, -90:-30]
X_test, Y_test = train[:1000, -180:-60], train[:1000, -60:]

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(200, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(100, return_sequences = False))
model.add(Dropout(0.2))
model.add(Dense(Y_train.shape[1]))
# model.compile(loss='mean_absolute_error', optimizer='adam')
model.compile(loss=smape, optimizer='adam')

# checkpoint
filepath="weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

model.fit(X_train, Y_train, epochs=50, batch_size=128, verbose=2, validation_data=(X_test, Y_test), callbacks=callbacks_list)


with open("lstm_model_50_epochs.pkl", 'w') as f:
    pickle.dump(model, f)

