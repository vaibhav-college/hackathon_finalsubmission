# LIBRARIES REQUIRED: Tensorflow, Numpy, Matplotlib.

# Utilize latest python and tensorflow version.

# The instructions to run this program are as followed:
# 1.) Install the required libraries in a python environment
# 2.) You can chose to load an AI model, or chose no to train an AI model. I have provided an AI model, for which you can
#     enter it's path into the terminal to load the AI model.
# 3.) It will auto predict, and when you close the matplotlib tab, it will give you a prompt to predict again. You can use
#     it as many times as you would like. If you stop predicting and press N, You will be asked to store the model for which
#     it is your discretion.

# This, is a CNN, also known as Convolutional Neural Network. It is better at image detection than a DNN (Deep Neural Network).
# The following AI model is fitted according to the dataset, as best as I can according to my knowledge.

# Thank you for considering me,
# Vaibhav Bhardwaj
# Project CNN Developer.

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random

from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Dropout
from keras.datasets import fashion_mnist

img_height = 28
img_width = 28

(train_img, train_lbl), (test_img, test_lbl) = fashion_mnist.load_data()

train_img, test_img = train_img / 255.0, test_img / 255.0

def plot_image(i, predictions_array, true_label, img):
  true_label, img = true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)  # type: ignore

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  true_label = true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

num_classes = 10
batchsize = 128

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

img_rows, img_cols = 28,28

loadmodel = input('Load model? (Y/N)  ')

if (loadmodel == 'y') or (loadmodel == 'Y'):
  
  path = input('Enter exact model folder path: ')

  path.replace('\\', '/')

  model = tf.keras.models.load_model(path)

else:
  if tf.keras.backend.image_data_format() == 'channels_first':
      train_img=train_img.reshape(train_img.shape[0],1,img_rows,img_cols)
      test_img =test_img.reshape(test_img.shape[0],1,img_rows,img_cols)
      input_shape=(1,img_rows,img_cols)
  else:
      train_img=train_img.reshape(train_img.shape[0],img_rows,img_cols,1)
      test_img =test_img.reshape(test_img.shape[0],img_rows,img_cols,1)
      input_shape=(img_rows,img_cols,1)

  model = Sequential([
      Conv2D(32, kernel_size=(5,5), activation='relu', input_shape=input_shape),
      Conv2D(64, kernel_size=(5,5), activation='relu'),
      # Conv2D(32, kernel_size=(3,3), activation='relu'),
      MaxPooling2D(pool_size=(6,6)),
      Flatten(),
      Dense(64, activation='relu'),
      Dropout(0.3),
      Dense(10, activation='relu')
  ])

  model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

  epoch = int(input('\n\nEpochs? (integer value):  '))

  model.fit(train_img, train_lbl, epochs=epoch, validation_data=(test_img, test_lbl), batch_size=batchsize)

predicter = 'y'

while (predicter == 'y') or (predicter == 'Y'):

    probability_model = tf.keras.Sequential([model, 
                                            tf.keras.layers.Softmax()])
    
    predictions = probability_model.predict(test_img)

    i = random.randrange(0,1857)
    plt.figure(figsize=(6,3))
    plt.subplot(1,2,1)
    plot_image(i, predictions[i], test_lbl, test_img)
    plt.subplot(1,2,2)
    plot_value_array(i, predictions[i],  test_lbl)
    plt.show()

    predicter = input('Predict? (Y/N)')

savemodel = input('Save model? (Y/N)')

if (savemodel == 'y') or (savemodel == 'Y'):

  direc = input('Please enter exact directory path: ')

  direc.replace('\\', '/')

  model.save(direc)   # type: ignore