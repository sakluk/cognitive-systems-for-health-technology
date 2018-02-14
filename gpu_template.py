# -*- coding: utf-8 -*-
"""
gpu_template.py
Cognitive Systems for Health Technology Applications
Sakari Lukkarinen & Juha Kopu, Feb 14, 2018

This script can be used for batch executions of your training experiments for
the Case 2. The code runs the model and saves the model and the history in
binary files.

Change the name of the code, model and history files so that that they are
unique and it is easy for you to recognize them when our GPU desktop operator
saves the results to local repository.

Otherwise you are free to try and test all the tricks you have learnt about
neural networks. For preprocessing, augmentation and modeling hints see 
the Kaggle competition winner for more details:
https://kaggle2.blob.core.windows.net/forum-message-attachments/88655/2795/competitionreport.pdf
"""

# Code, model and history filenames
my_code = 'gpu_template.py'
model_filename = 'case_2_model.h5'
history_filename = 'case_2_history.p'

# Info for the operator
import time
print('----------------------------------------------------------------------')
print(' ')
print('Starting the code (', time.asctime(), '):', my_code)
print(' ')

# Import libraries and functions
#import numpy as np
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
batch_size = 10
epochs = 10
steps_per_epoch = 10 
validation_steps = 10
image_height = 150
image_width = 150 

# Create datagenerators
# All images are rescaled by 1/255 in addition:
# Zooming and horizontal flip is used for train dataset
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
validation_datagen = ImageDataGenerator(rescale = 1./255)
test_datagen = ImageDataGenerator(rescale = 1./255)

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
model.add(layers.Conv2D(16, (3, 3), activation = 'relu', 
                        input_shape = (image_height, image_width, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32, (3, 3), activation = 'relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation = 'relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))


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


