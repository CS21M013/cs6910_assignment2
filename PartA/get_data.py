import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Input, InputLayer, Flatten, Conv2D, BatchNormalization, MaxPooling2D, Activation , GlobalAveragePooling2D
from tensorflow.keras.models import Sequential,  Model
import tensorflow.keras.backend as K
import wandb
import tensorflow as tf
from tensorflow import keras
from sklearn import metrics
import pandas as pd
import cv2
from wandb.keras import WandbCallback
import pathlib
import seaborn as sns
from paths import*
'''
Function to get the training data and validation data for training
'''
def get_data(batch_size,data_augmentation,train_data_path):

    # Boolean variable specifying whether we are doing data augmentation or not
    data_augmentation = data_augmentation
    # Bactch size for training
    BATCH_SIZE = batch_size

    '''
    Performing data augemtation if true as a hyper parameter
    '''
    if data_augmentation == True:
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                rescale=1./255, # rescale the images
                validation_split = 0.1, # creating the validation data from training data as 10% of training data
                shear_range=0.2, # shearing range = 0.2
                zoom_range=0.2, # zoom range = 0.2
                rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
                width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
                height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
                horizontal_flip=True,  # randomly flip images
                vertical_flip=False # Not alloewing the images to be vertically flipped
                )
    # If data augmentation is not performed, we are just generating the training data validation data as a 10% of the training data.
    else:
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,validation_split = 0.1)

    '''
    Generating the training data batch wise
    '''
    train_generator = train_datagen.flow_from_directory(
        train_data_path,
        subset='training',
        target_size=img_size,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle = True,
        seed = 123)
        
    '''
    Generating the validation data batch wise
    '''
    validation_generator = train_datagen.flow_from_directory(
        train_data_path,
        target_size=img_size,
        subset = 'validation',
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle = True,
        seed = 123)

    return train_generator,validation_generator