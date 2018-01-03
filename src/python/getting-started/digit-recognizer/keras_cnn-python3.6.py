#!/usr/bin/python
# coding: utf-8
'''
Created on 2017-12-26
Update  on 2017-12-26
Author: xiaomingnio
Github: https://github.com/apachecn/kaggle
'''

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from keras.callbacks import ReduceLROnPlateau
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPool2D
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical  # convert to one-hot-encoding

np.random.seed(2)

# Load the data
train = pd.read_csv(
    r'datasets/getting-started/digit-recognizer/input/train.csv')
test = pd.read_csv(r'datasets/getting-started/digit-recognizer/input/test.csv')

X_train = train.values[:, 1:]
Y_train = train.values[:, 0]
test = test.values

# Normalization
X_train = X_train / 255.0
test = test / 255.0

# Reshape image in 3 dimensions (height = 28px, width = 28px , canal = 1)
X_train = X_train.reshape(-1, 28, 28, 1)
test = test.reshape(-1, 28, 28, 1)

# Encode labels to one hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])
Y_train = to_categorical(Y_train, num_classes=10)

# Set the random seed
random_seed = 2

# Split the train and the validation set for the fitting
X_train, X_val, Y_train, Y_val = train_test_split(
    X_train, Y_train, test_size=0.1, random_state=random_seed)

# Set the CNN model 
# my CNN architechture is In -> [[Conv2D->relu]*2 -> MaxPool2D -> Dropout]*2 -> Flatten -> Dense -> Dropout -> Out

model = Sequential()

model.add(
    Conv2D(
        filters=32,
        kernel_size=(5, 5),
        padding='Same',
        activation='relu',
        input_shape=(28, 28, 1)))
model.add(
    Conv2D(
        filters=32, kernel_size=(5, 5), padding='Same', activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(
    Conv2D(
        filters=64, kernel_size=(3, 3), padding='Same', activation='relu'))
model.add(
    Conv2D(
        filters=64, kernel_size=(3, 3), padding='Same', activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation="softmax"))

# Define the optimizer
optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

# Compile the model
model.compile(
    optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

epochs = 30
batch_size = 86

# Set a learning rate annealer
learning_rate_reduction = ReduceLROnPlateau(
    monitor='val_acc', patience=3, verbose=1, factor=0.5, min_lr=0.00001)

datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
    zoom_range=0.1,  # Randomly zoom image 
    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=False,  # randomly flip images
    vertical_flip=False)  # randomly flip images

datagen.fit(X_train)

history = model.fit_generator(
    datagen.flow(
        X_train, Y_train, batch_size=batch_size),
    epochs=epochs,
    validation_data=(X_val, Y_val),
    verbose=2,
    steps_per_epoch=X_train.shape[0] // batch_size,
    callbacks=[learning_rate_reduction])

# predict results
results = model.predict(test)

# select the indix with the maximum probability
results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("datasets/getting-started/digit-recognizer/ouput/Result_keras_CNN.csv",index=False)