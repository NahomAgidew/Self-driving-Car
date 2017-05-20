"""
THE AI

Author: Nahom Abi
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from utils import INPUT_SHAPE, batch_generator
import argparse
import os

np.random.seed(0)

# center, left, right, steering, speed, throttle, brake
# df.get_values()[:, :3]
# df.get_values()[:, 4:5]
data_df = pd.read_csv('./driving_log.csv')
X = data_df.get_values()[:, :3]
y = data_df.get_values()[:, 4:5]

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.20, random_state=0)

"""
    NVIDIA model used
    Image normalization to avoid saturation and make gradients work better.
    Convolution: 5x5, filter: 24, strides: 2x2, activation: ELU
    Convolution: 5x5, filter: 36, strides: 2x2, activation: ELU
    Convolution: 5x5, filter: 48, strides: 2x2, activation: ELU
    Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
    Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
    Drop out (0.5)
    Fully connected: neurons: 100, activation: ELU
    Fully connected: neurons: 50, activation: ELU
    Fully connected: neurons: 10, activation: ELU
    Fully connected: neurons: 1 (output)
    the convolution layers are meant to handle feature engineering
    the fully connected layer for predicting the steering angle.
    dropout avoids overfitting
    ELU(Exponential linear unit) function takes care of the Vanishing gradient problem. 
"""
model = Sequential()
model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=INPUT_SHAPE))
model.add(Conv2D(24, 5, 5, activation='elu', subsample=(2, 2)))
model.add(Conv2D(36, 5, 5, activation='elu', subsample=(2, 2)))
model.add(Conv2D(48, 5, 5, activation='elu', subsample=(2, 2)))
model.add(Conv2D(64, 3, 3, activation='elu'))
model.add(Conv2D(64, 3, 3, activation='elu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100, activation='elu'))
model.add(Dense(50, activation='elu'))
model.add(Dense(10, activation='elu'))
model.add(Dense(1))
model.summary()

checkpoint = ModelCheckpoint('model-{epoch:03d}.h5', monitor='val_loss', verbose=0, save_best_only=True, mode='auto')
model.compile(loss='mean_squared_error', optimizer=Adam(lr=1e-4))
# able us to do real-time image augementation on the CPU in parallel to training the model on the GPU
model.fit_generator(batch_generator('data', X_train, y_train, 40, True),
                    20000,
                    10,
                    max_q_size=1,
                    validation_data=batch_generator('data', X_valid, y_valid, 40, False),
                    nb_val_samples=len(X_valid),
                    callbacks=[checkpoint],
                    verbose=1)
