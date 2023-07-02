#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, Dropout
from keras.layers import Reshape
from keras.layers import LSTM


# Read the data from the CSV file
data = pd.read_csv('data.csv')
data = data.drop(['filename'], axis=1)

# Preprocess the data
scaler = StandardScaler()
X = scaler.fit_transform(np.array(data.iloc[:, :-1], dtype=float))
y = data.iloc[:, -1]

# Split the dataset into training and testing sets
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2)
train_x = np.reshape(train_x, (train_x.shape[0], 13, 2, 1))
test_x = np.reshape(test_x, (test_x.shape[0], 13, 2, 1))

# Build the neural network model with LSTM
model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(13, 2)))
model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256, return_sequences=True))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(512, activation="relu"))
model.add(Dense(13*2*1, activation="linear"))  # Use linear activation for EQ emulation

# Reshape the output to match the input shape
model.add(Reshape((13, 2, 1)))

# Compile and train the model
model.compile(optimizer="adam", loss="mean_squared_error")
model.fit(train_x, train_x, batch_size=64, epochs=100, validation_data=(test_x, test_x))