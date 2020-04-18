from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import os
import random
import numpy as np
import cv2
import pandas as pd
import math
import re
batch_size = 10
num_classes = 10
epochs = 50

# input image dimensions

# the data, split between train and test sets
#(x_train, y_train), (x_test, y_test) = mnist.load_data()
"""
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
"""
dataset_train= pd.read_csv('binding_train.tsv',sep='\t',header=0, usecols=[1,37,8], dtype = {8:str})
print("Train imported")
dataset_test = pd.read_csv('binding_test.tsv',sep='\t',header=0, usecols=[1,37,8])
print("Test imported")
x_train = []
y_train = []
x_test = []
y_test = []
print (dataset_train.head())
x_train = np.array(dataset_train.iloc[:,[0,2]])
x_test = np.array(dataset_test.iloc[:,[0,2]])
bindingdatatrain = np.array(dataset_train.iloc[:,1])
print(bindingdatatrain)
print("arrays done")
for ki in bindingdatatrain:
    ki = str(ki)
    ki= re.sub(r'[^\d.]', '', ki)
    if ki=="":
        ki = 'nan'
    ki = float(ki)
    if math.isnan(ki):
        y_train.append(10)
    else:
        if ki<1:
            y_train.append(1)
        elif ki<5:
            y_train.append(2)
        elif ki<10:
            y_train.append(3)
        elif ki<50:
            y_train.append(4)
        elif ki<100:
            y_train.append(5)
        elif ki<500:
            y_train.append(6)
        elif ki<1000:
            y_train.append(7)
        elif ki<5000:
            y_train.append(8)
        else:
            y_train.append(9)
bindingdatatest = np.array(dataset_test.iloc[:,1])
for ki in bindingdatatest:
    ki = float(ki)
    if math.isnan(ki):
        y_test.append(10)
    else:
        if ki<1:
            y_test.append(1)
        elif ki<5:
            y_test.append(2)
        elif ki<10:
            y_test.append(3)
        elif ki<50:
            y_test.append(4)
        elif ki<100:
            y_test.append(5)
        elif ki<500:
            y_test.append(6)
        elif ki<1000:
            y_test.append(7)
        elif ki<5000:
            y_test.append(8)
        else:
            y_test.append(9)
y_train = np.asarray(y_train)
y_test = np.asarray(y_test)
print ("classification done")

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
input_shape=(x_train.shape[1],1)
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
model.save('cnn.h5')
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
