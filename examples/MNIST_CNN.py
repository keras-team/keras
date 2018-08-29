#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 23:22:02 2018

@author: kaivanshah
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns

np.random.seed(2)

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools

from keras.utils.np_utils import to_categorical 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau


sns.set(style='white', context='notebook', palette='deep')

train = pd.read_csv('HandWrittenTrain.csv')
test = pd.read_csv('Hand_written_test.csv')

X_train = train.drop("label",axis=1)
Y_train = train["label"]
X_test  = test.copy()

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
scaler.fit(X_test)
X_test = scaler.transform(X_test)

X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28,1)

from keras.utils import to_categorical
Y_train = to_categorical(Y_train, num_classes= 10)
random_seed = 2
# Split the train and the validation set for the fitting
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=random_seed)
#y_test_oneHot = to_categorical(Y_test, num_classes=10)

from keras.layers import Convolution2D, Dense, MaxPooling2D, Dropout, Flatten
from keras.models import Sequential

model = Sequential()
model.add(Convolution2D(filters=32, kernel_size=(5, 5), padding='same', input_shape = (28, 28, 1), activation='relu'))
model.add(Convolution2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu'))       
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

model.add(Convolution2D(filters=64, kernel_size=(5, 5), padding='same', activation='relu'))       
model.add(Convolution2D(filters=64, kernel_size=(5, 5), padding='same', activation='relu'))       
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(units=256, activation='relu',))
model.add(Dropout(0.3))
model.add(Dense(units=10, activation='softmax'))


model.compile(optimizer='RMSprop', loss = 'categorical_crossentropy', metrics=['accuracy'])

from keras.callbacks import ReduceLROnPlateau
# To keep the advantage of the fast computation time with a high LR, i decreased the LR dynamically every X steps (epochs) depending if it is necessary (when accuracy is not improved).
# With the ReduceLROnPlateau function from Keras.callbacks, i choose to reduce the LR by half if the accuracy is not improved after 3 epochs.

# Set a learning rate annealer
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.000001)


from keras.preprocessing.image import ImageDataGenerator
# I did not apply a vertical_flip nor horizontal_flip since it could have lead to misclassify symetrical numbers such as 6 and 9.

datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images
datagen.fit(X_train)

history = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=512),
                              epochs = 30, validation_data=(X_val, Y_val),
                              verbose = 1, steps_per_epoch=X_train.shape[0]/512, 
                              callbacks=[learning_rate_reduction])


results = model.predict(X_test)
# select the indix with the maximum probability
results = np.argmax(results,axis = 1)
results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)
submission.to_csv("submission.csv",index=False)


