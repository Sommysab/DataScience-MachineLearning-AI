#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv('Car_Purchasing_Data.csv', encoding='ISO-8859-1')

X = df.drop(
      [
        'Customer Name', 
        'Customer e-mail', 
        'Country', 
        'Car Purchase Amount'
      ], 
      axis=1
    )

y = df['Car Purchase Amount']


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_scaled = scaler.fit_transform(X)

y_scaled = scaler.fit_transform(y.values.reshape(-1,1))


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.25)


import tensorflow.keras
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()

model.add(Dense(25, input_dim = 5, activation = 'relu'))

model.add(Dense(25, activation='relu'))

model.add(Dense(1, activation = 'linear'))

model.compile(optimizer='adam', loss='mean_squared_error')

epochs_hist = model.fit(X_train, y_train, epochs=1, batch_size=50, verbose=1, validation_split=0.2)


plt.plot(epochs_hist.history['loss'])

plt.plot(epochs_hist.history['val_loss'])

plt.title('Model Loss Progress During training')

plt.ylabel('Training and Validation Loss')

plt.xlabel('Epoch number')

plt.legend(['Training loss', 'Validation loss'])


# Single Client
X_test = np.array([[1,50,50000,10000,600000]])

y_predict = model.predict(X_test)

# Predicted Afford Amount
print('Predicted Purchase Amount is', y_predict)

