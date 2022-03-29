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
Function to get the test data in batches
'''
def get_test_data(batch_size,test_data_path,shuffle=True):
    # Using image data generator of keras to get test data in batches
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
                test_data_path,
                target_size=(img_width,img_height),
                batch_size=batch_size,
                class_mode='categorical',
                shuffle = True, seed=1234)
    
    return test_generator