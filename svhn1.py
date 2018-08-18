#first import the dataset and convert it to greyscale
import scipy.io
Train = scipy.io.loadmat('datasets/3232train/train_32x32.mat')
Test = scipy.io.loadmat('datasets/3232test/test_32x32.mat')

import numpy as np
X_train = Train['X']
y_train = Train['y']
X_test = Test['X']
y_test = Test['y']

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

print(X_train.shape)

X_train = X_train[np.newaxis,...]
X_train = np.swapaxes(X_train,0,4).squeeze()

X_test = X_test[np.newaxis,...]
X_test = np.swapaxes(X_test,0,4).squeeze()


def rgb2gray(images):
    return np.expand_dims(np.dot(images, [0.2990, 0.5870, 0.1140]), axis=3)

train_greyscale = rgb2gray(X_train).astype(np.float32)
test_greyscale = rgb2gray(X_test).astype(np.float32)

del X_train,X_test
print("Training Set", train_greyscale.shape)
print("Test Set", test_greyscale.shape)

print(y_train.shape)
print(y_test.shape)

np.place(y_train,y_train == 10,0)
np.place(y_test,y_test == 10,0)

from keras.utils import to_categorical
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)


#create a convolutional neural network
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout,Activation,BatchNormalization

classifier = Sequential()

classifier.add(Convolution2D(32, (3,3) , input_shape = (32,32,1), padding='same', activation = 'relu'))
classifier.add(BatchNormalization())
classifier.add(Convolution2D(32, (3, 3) , activation = 'relu'))
classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Dropout(0.25))
classifier.add(Convolution2D(64, (3, 3) , padding='same', activation ='relu'))
classifier.add(BatchNormalization())
classifier.add(Convolution2D(64, (3, 3) , activation ='relu'))
classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Dropout(0.25))
classifier.add(Flatten())
classifier.add(BatchNormalization())
classifier.add(Dense(512))
classifier.add(Activation('relu'))
classifier.add(BatchNormalization())
classifier.add(Dropout(0.5))
classifier.add(Dense(10))
classifier.add(Activation('softmax'))

from keras.optimizers import SGD
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
classifier.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])

#train the CNN using the dataset
import tensorflow as tf
with tf.device('/gpu:0'):
    classifier.fit(train_greyscale, y_train, batch_size=128, nb_epoch=10, verbose=1, validation_data=(test_greyscale, y_test))
    
score = classifier.evaluate(test_greyscale, y_test, verbose=0)
print('loss:', score[0])
print('Test accuracy:', score[1])
