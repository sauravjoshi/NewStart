# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 23:51:38 2018

@author: sj250305
"""
from keras.datasets import mnist
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

#Baseline Model with Multi-Layer Perceptrons

import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils

seed = 7
numpy.random.seed(seed)
"""
For a multi-layer perceptron model we must reduce the images down into a vector
of pixels.
In this case the 28×28 sized images will be 784 pixel input values.
"""
labeled_images = pd.read_csv('input/train.csv')
images = labeled_images.iloc[:,1:]
labels = labeled_images.iloc[:,:1]

train_images, test_images,train_labels, test_labels = train_test_split(images, labels, train_size=0.8, random_state=0)


train_images = train_images.astype('float32')
test_images = test_images.astype('float32')

train_images = train_images / 255
test_images = test_images / 255

train_labels = np_utils.to_categorical(train_labels)
test_labels = np_utils.to_categorical(test_labels)
num_classes = test_labels.shape[1]    

num_pixels = 784

def baseline_model():
    model = Sequential()
    model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal'
              , activation='relu'))
    model.add(Dense(num_classes, kernel_initializer='normal'
                    ,activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='adam',
                  metrics=['accuracy'])
    return model

# build the model
model = baseline_model()
# Fit the model
model.fit(train_images, train_labels, validation_data=(test_images, test_labels),
          epochs=15, batch_size=500, verbose=2)
# Final evaluation of the model
scores = model.evaluate(test_images, test_labels, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))
#Baseline Error: 2.24%, 2.07%
#Accuracy: 97.76, 97.93%

test_data = pd.read_csv('input/test.csv')
results = model.predict(test_data)

#Decoding the encodede Categorical Variable to give a single Value
inverted = [numpy.argmax(r,axis=None, out=None)for r in results]
df = pd.DataFrame(inverted)
df.index.name='ImageId'
df.index+=1
df.columns=['Label']
df.to_csv('results.csv', header=True)