# import libraries
import tensorflow as tf
import torch
from tensorflow import keras
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from PIL import Image

#Loading data from Internet
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# check data shape
np.shape(x_train)

# Data Visualization
fig, ax = plt.subplots(2,5)
for i, ax in enumerate(ax.flatten()):
    im_idx = np.argwhere(y_train == i)[0]
    plottable_image = np.reshape(x_train[im_idx], (28, 28))
    ax.imshow(plottable_image, cmap='plasma')

# Reshaping the array to 4-dims so that it can work with the Keras API
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)
# Making sure that the values are float so that we can get decimal points after division
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# Normalizing the RGB codes by dividing it to the max RGB value.
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print('Number of images in x_train', x_train.shape[0])
print('Number of images in x_test', x_test.shape[0])

# Importing the required Keras modules containing model and layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
# Creating a Sequential Model and adding the layers
model = Sequential()
model.add(Conv2D(28, kernel_size=(3,3), input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
model.add(Dense(128, activation=tf.nn.relu))
model.add(Dropout(0.20))
model.add(Dense(16, activation=tf.nn.relu))
model.add(Dense(10,activation=tf.nn.softmax))

# ## 5. Compile the model

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])
history = model.fit(x=x_train,y=y_train, validation_data=(x_test, y_test), epochs=10, verbose=0)

# evaluate the model
_, train_acc = model.evaluate(x_train, y_train, verbose=0)
_, test_acc = model.evaluate(x_test, y_test, verbose=0)
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))

# multilayer perceptron model for the two circles problem
from sklearn.datasets import make_circles
from keras.models import Sequential
from keras.layers import Dense
from matplotlib import pyplot

# plot loss during training
pyplot.subplot(211)
pyplot.title('Loss')
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
# plot accuracy during training
pyplot.subplot(212)
pyplot.title('Accuracy')
pyplot.plot(history.history['acc'], label='train')
pyplot.plot(history.history['val_acc'], label='test')
pyplot.legend()
pyplot.show()

# demonstration of calculating metrics for a neural network model using sklearn
from sklearn.datasets import make_circles
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense

# predict probabilities for test set
yhat_probs = model.predict(x_test, verbose=0)
# predict crisp classes for test set
yhat_classes = model.predict_classes(x_test, verbose=0)

# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(y_test, yhat_classes)
print('Accuracy: %f' % accuracy)
# precision tp / (tp + fp)
precision = precision_score(y_test, yhat_classes,average='micro')
print('Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(y_test, yhat_classes,average='micro')
print('Recall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(y_test, yhat_classes,average='micro')
print('F1 score: %f' % f1)

# kappa
kappa = cohen_kappa_score(y_test, yhat_classes)
print('Cohens kappa: %f' % kappa)
# confusion matrix
matrix = confusion_matrix(y_test, yhat_classes)
print(matrix)

####################################################
# # Noisy MNIST

# import libraries
from io import StringIO,BytesIO

##Loading data from Internet
#Unziping data to lacal disk
import requests, zipfile
r = requests.get("http://www.iro.umontreal.ca/~lisa/icml2007data/mnist_background_random.zip", stream=True)
z = zipfile.ZipFile(BytesIO(r.content))
z.extractall()

import numpy as np
#Assignig datas to variables
b_data = np.loadtxt('mnist_background_random_test.amat')
a_data = np.loadtxt('mnist_background_random_train.amat')

# get train image datas
ax_train = a_data[:, :-1] / 1.0

# get train image labels
ay_train = a_data[:, -1:]

# get test image datas
bx_test = b_data[:, :-1] / 1.0

# get test image labels
by_test = b_data[:, -1:]

ax_train = ax_train.reshape(-1,28, 28)
bx_test = bx_test.reshape(-1,28, 28)

# Data Visualization
fig, ax = plt.subplots(2,5)
for i, ax in enumerate(ax.flatten()):
    im_idx = np.argwhere(y_train == i)[0]
    plottable_image = np.reshape(ax_train[im_idx], (28, 28))
    ax.imshow(plottable_image, cmap='afmhot_r')

# Reshaping the array to 4-dims so that it can work with the Keras API
ax_train = ax_train.reshape(ax_train.shape[0], 28, 28, 1)
bx_test = bx_test.reshape(bx_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)
# Making sure that the values are float so that we can get decimal points after division
ax_train = ax_train.astype('float32')
bx_test = bx_test.astype('float32')
# Normalizing the RGB codes by dividing it to the max RGB value.
ax_train /= 255
bx_test /= 255
print('x_train shape:', ax_train.shape)
print('Number of images in ax_train', ax_train.shape[0])
print('Number of images in bx_test', bx_test.shape[0])

# Importing the required Keras modules containing model and layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
# Creating a Sequential Model and adding the layers
model = Sequential()
model.add(Conv2D(28, kernel_size=(3,3), input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
model.add(Dense(128, activation=tf.nn.relu))
model.add(Dropout(0.20))
model.add(Dense(32, activation=tf.nn.relu))
model.add(Dense(64, activation=tf.nn.relu))
model.add(Dense(32, activation=tf.nn.relu))
model.add(Dropout(0.20))
model.add(Dense(10,activation=tf.nn.softmax))

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

history = model.fit(x=ax_train,y=ay_train, validation_data=(bx_test, by_test), epochs=17, verbose=0)

# evaluate the model
_, train_acc = model.evaluate(ax_train, ay_train, verbose=0)
_, test_acc = model.evaluate(ax_test, by_test, verbose=0)
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))

# plot loss during training
pyplot.subplot(211)
pyplot.title('Loss')
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
# plot accuracy during training
pyplot.subplot(212)
pyplot.title('Accuracy')
pyplot.plot(history.history['acc'], label='train')
pyplot.plot(history.history['val_acc'], label='test')
pyplot.legend()
pyplot.show()

# predict probabilities for test set
yxhat_probs = model.predict(ax_test, verbose=0)
# predict crisp classes for test set
yxhat_classes = model.predict_classes(ax_test, verbose=0)

# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(ay_test, yxhat_classes)
print('Accuracy: %f' % accuracy)
# precision tp / (tp + fp)
precision = precision_score(ay_test, yxhat_classes,average='micro')
print('Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(ay_test, yxhat_classes,average='micro')
print('Recall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(ay_test, yxhat_classes,average='micro')
print('F1 score: %f' % f1)

# kappa
kappa = cohen_kappa_score(ay_test, yxhat_classes)
print('Cohens kappa: %f' % kappa)
# confusion matrix
matrix = confusion_matrix(ay_test, yxhat_classes)
print(matrix)


# # Zaman ayırdığınız için şimdiden teşekkür ederiz. Herhangi bir yardımda bulunabilirsem, haberim olsun.

# # saygılarımla
# Ali Saadat
