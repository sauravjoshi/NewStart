# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 01:45:30 2018

@author: sj250305
"""

import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')

seed = 7
numpy.random.seed(seed)

import pandas as pd
labeled_images = pd.read_csv('input/train.csv')
images = labeled_images.iloc[:,1:]
labels = labeled_images.iloc[:,:1]

from sklearn.model_selection import train_test_split
train_images, test_images,train_labels, test_labels = train_test_split(images, labels, train_size=0.8, random_state=0)

#Converting the dataframe to ndarray, so that reshaping can be done.
train_images = train_images.as_matrix()
test_images = test_images.as_matrix()
train_labels = train_labels.as_matrix()
test_labels = test_labels.as_matrix()

train_images = train_images.reshape(train_images.shape[0],1,28,28).astype('float32')
test_images = test_images.reshape(test_images.shape[0],1,28,28).astype('float32')


train_images = train_images / 255
test_images = test_images / 255

train_labels = np_utils.to_categorical(train_labels)
test_labels = np_utils.to_categorical(test_labels)
num_classes = test_labels.shape[1] 

num_pixels = 784

def baseline_model():
	# create model
	model = Sequential()
	model.add(Conv2D(32, (5, 5), input_shape=(1, 28, 28), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dense(num_classes, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

model = baseline_model()
# Fit the model
model.fit(train_images, train_labels, validation_data=(test_images, test_labels), epochs=10, batch_size=200, verbose=2)
# Final evaluation of the model
scores = model.evaluate(test_images, test_labels, verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))

test_data = pd.read_csv('input/test.csv')
test_data = test_data.as_matrix()
test_data = test_data.reshape(test_data.shape[0],1,28,28).astype('float32')

results = model.predict(test_data)
#Decoding the encodede Categorical Variable to give a single Value
inverted = [numpy.argmax(r,axis=None, out=None)for r in results]
df = pd.DataFrame(inverted)
df.index.name='ImageId'
df.index+=1
df.columns=['Label']
df.to_csv('results_cnn.csv', header=True)

