import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense


def build_gru(input_shape):
    model = Sequential()
    model.add(GRU(units=100, return_sequences=False, input_shape=input_shape))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model
