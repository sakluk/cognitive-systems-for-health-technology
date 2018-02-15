# -*- coding: utf-8 -*-
"""
Lukkarinen_Case_2_Run_01.py
Cognitive Systems for Health Technology Applications
Sakari Lukkarinen & Juha Kopu, Feb 15, 2018
"""

# Code, model and history filenames
my_code = 'Lukkarinen_Case_2_Run_01.py'
model_filename = 'Lukkarinen_Case_2_Run_01.h5'
history_filename = 'Lukkarinen_Case_2_Run_01.p'

# Info for the operator
import time
print('----------------------------------------------------------------------')
print(' ')
print('Starting the code (', time.asctime(), '):', my_code)
print(' ')

# Import libraries and functions
import numpy as np
import pickle
from keras import layers
from keras import models
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
import os

# Dataset directories and labels files
train_dir = "../dataset2/train" 
validation_dir = "../dataset2/validation" 
test_dir = "../dataset2/test" 
#train_dir = "../fullset/train"
#validation_dir = "../fullset/test"
#test_dir = "../fullset/test"
#train_labels_file = '../trainLabels.csv'
#test_labels_file = '../retinopathy_solution.csv'

# Training parameters
batch_size = 32
epochs = 10
steps_per_epoch = 39 
validation_steps = 13
image_height = 150
image_width = 200 

# My own preprocessing function
def my_fun(x):
    r = x[:, :, 0]
    g = x[:, :, 1]
    b = x[:, :, 2]
    r = (r - np.mean(r))/(12.0*np.std(r))
    g = (g - np.mean(g))/(12.0*np.std(g))
    b = (b - np.mean(g))/(12.0*np.std(b))
    x[:, :, 0] = r
    x[:, :, 1] = g
    x[:, :, 2] = b
    x += 0.5
    return x

# Create datagenerators
# All images are rescaled by 1/255 in addition:
# Zooming and horizontal flip is used for train dataset
train_datagen = ImageDataGenerator(preprocessing_function = my_fun)
validation_datagen = ImageDataGenerator(preprocessing_function = my_fun)
test_datagen = ImageDataGenerator(preprocessing_function = my_fun)

# Generator for train dataset
print('Training dataset.')
train_generator = train_datagen.flow_from_directory(
        # This is the target directory
        train_dir,
        # All images will be resized to 150x150
        target_size = (image_height, image_width),
        batch_size = batch_size,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode = 'binary')

# Generator for validation dataset
print('Validation dataset.')
validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size = (image_height, image_width),
        batch_size = batch_size,
        class_mode = 'binary')

# Generator for test dataset
print('Test dataset.')
test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size = (image_height, image_width),
        batch_size = batch_size,
        class_mode = 'binary')


# Build the model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation = 'relu', 
                        input_shape = (image_height, image_width, 3)))
model.add(layers.Conv2D(32, (3, 3), activation = 'relu'))
model.add(layers.MaxPool2D((3, 3), strides=2))

model.add(layers.Conv2D(64, (3, 3), activation = 'relu'))
model.add(layers.Conv2D(64, (3, 3), activation = 'relu'))
model.add(layers.MaxPool2D((3, 3), strides=2))

model.add(layers.Conv2D(96, (3, 3), activation = 'relu'))
model.add(layers.Conv2D(96, (3, 3), activation = 'relu'))
model.add(layers.MaxPool2D((3, 3), strides=2))

model.add(layers.Conv2D(128, (3, 3), activation = 'relu'))
model.add(layers.Conv2D(128, (3, 3), activation = 'relu'))
#model.add(layers.MaxPool2D((2, 2)))

model.add(layers.Conv2D(160, (3, 3), activation = 'relu'))
model.add(layers.Conv2D(160, (3, 3), activation = 'relu'))
#model.add(layers.MaxPool2D((3, 3), strides=1))
model.add(layers.Dropout(0.1))

model.add(layers.Conv2D(192, (3, 3), activation = 'relu'))
model.add(layers.Conv2D(192, (3, 3), activation = 'relu'))
#model.add(layers.MaxPool2D((3, 3), strides=2))
model.add(layers.Dropout(0.1))

model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(1, activation='sigmoid'))

model.summary()

# Compile the model
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(),
              metrics=['acc'])


# Train the model and watch the elapsed time
t1 = time.time()
h = model.fit_generator(
      train_generator,
      steps_per_epoch = steps_per_epoch,
      verbose = 1,
      epochs = epochs,
      validation_data = validation_generator,
      validation_steps = validation_steps)
t2 = time.time()

# Store the elapsed time into history
h.history.update({'time_elapsed': t2 - t1})
print(' ')
print('Total elapsed time for training: {:.3f} minutes'.format((t2-t1)/60))
print(' ')

# Save the model
print('Model is saved to file:', model_filename)
fname = os.path.join('..', 'Results', model_filename)
model.save(fname)
# Can be loaded back with commands:
# from keras.models import load_model
# model = load_model(model_filename) 

# Save the history
print('History is saved to file:', history_filename)
fname = os.path.join('..', 'Results', history_filename)
pickle.dump(h.history, open(fname, 'wb'))
# Can be loaded back with command:
# h2 = pickle.load(open(history_filename, 'rb'))

# Info for the operator
print(' ')
print('Done (', time.asctime(), '):', my_code)
print('----------------------------------------------------------------------')
print(' ')


