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

class cnnModel():

    '''
    Initialization of the model hyperparameters for executing the Wandb Sweep and hyperparameter tuning
    '''
    def __init__(self,img_size,model_parameters,num_classes):
        self.num_hidden_cnn_layer = model_parameters["num_hidden_cnn_layer"]
        self.activation = model_parameters["activation"]
        self.batch_normalization = model_parameters["batch_normalization"]
        self.filter_multiplier = model_parameters["filter_multiplier"]
        self.filter_size = model_parameters["filter_size"]
        self.num_filters = model_parameters["num_filters"]
        self.pool_size=model_parameters["pool_size"]
        self.dropout = model_parameters["dropout"]
        self.padding = model_parameters["padding"]
        self.optimizer = model_parameters["optimizer"]
        self.gap = model_parameters["gap"]
        self.batchnorm_location = model_parameters["batchnorm_location"]
        self.dense_neurons = model_parameters["dense_neurons"]
        self.num_classes = num_classes
        self.input_shape = (img_size[0],img_size[1],3)
        self.dropout_loc = model_parameters["dropout_loc"]


    '''
    Single CNN block including 2D Convolutional layers, Batch normalization, either before or after activation and max pooling layer
    '''
    def cnnBlock(self,model,i):
        # Standard filter distribution - same number of filters in all Convolutional layers
        if self.filter_multiplier == "standard":
            model.add(Conv2D(self.num_filters, self.filter_size,kernel_initializer = "he_uniform",padding = self.padding))
    
        # Double filter distribution - double number of filters in each Convolutional layers
        elif self.filter_multiplier == "double":
            model.add(Conv2D(2**(i+1)*self.num_filters, self.filter_size,kernel_initializer = "he_uniform", padding = self.padding))
    
        # Halve the filter size in each successive convolutional layers
        elif self.filter_multiplier == "half":
            model.add(Conv2D(int(self.num_filters/2**(i+1)), self.filter_size,kernel_initializer = "he_uniform", padding = self.padding))
    
        # Batch normalization before activation layer
        if self.batchnorm_location == "Before" and self.batch_normalization: model.add(BatchNormalization())
        model.add(Activation(self.activation))
    
        # Batch normalization after activation layer
        if self.batchnorm_location == "After" and self.batch_normalization: model.add(BatchNormalization())
    
        # Max pooling layer
        model.add(MaxPooling2D(pool_size=self.pool_size))

        return model
        

    '''
    Building the model using the CNN blocks
    '''
    def build_cnnmodel(self):
        # Including the GPU computation
        with tf.device('/device:GPU:0'):
            tf.keras.backend.clear_session()

            # Sequential model
            model = Sequential()
            
            #First CNN layer connecting to input layer
            model.add(Conv2D(self.num_filters, self.filter_size, padding = self.padding,kernel_initializer = "he_uniform", input_shape = self.input_shape))
            # Batch normalization before the activation
            if self.batchnorm_location == "Before" and self.batch_normalization: model.add(BatchNormalization())
            model.add(Activation(self.activation))
            # Batch normalization after the activation
            if self.batchnorm_location == "After" and self.batch_normalization: model.add(BatchNormalization())
            # Max pooling layer
            model.add(MaxPooling2D(pool_size=self.pool_size))  
             
            # Dropout after convolutional layers or all the layers 
            if self.dropout_loc == "conv" or self.dropout_loc=="all":
                # Checking if value is not NONE
                if self.dropout != None:
                    model.add(tf.keras.layers.Dropout(self.dropout))
                # adding the CNN blocks and the dropout after them if allowed
                for i in range(self.num_hidden_cnn_layer-1):
                    # CNN block adding
                    model = self.cnnBlock(model,i)
                    # Adding dropout
                    if self.dropout != None:
                        model.add(tf.keras.layers.Dropout(self.dropout))

            elif self.dropout_loc == "dense":
                # adding the CNN blocks and no dropout as dropout only for dense
                for i in range(self.num_hidden_cnn_layer-1):
                    model = self.cnnBlock(model,i)

            
            # Either Flattening  or Global average pooling
            if self.gap == True:
                # adding Global average pooling
                model.add(GlobalAveragePooling2D())
            else: 
                # Adding flatten layer
                model.add(Flatten())

            # Adding dropout after dense layer if dropout location is dense or all
            if self.dropout_loc == "dense" or self.dropout_loc =="all":
                # adding dense layer
                model.add(Dense(self.dense_neurons, activation = 'sigmoid'))
                if self.dropout != None:
                    # dropout layer after dense layer
                    model.add(tf.keras.layers.Dropout(self.dropout))
                # Final dens eoutput layer
                model.add(Dense(self.num_classes, activation = 'softmax'))

            # only dense followed by final out put layer and no dropout layer
            elif self.dropout_loc =="conv":
                model.add(Dense(self.dense_neurons, activation = 'sigmoid'))
                # final dense output layer
                model.add(Dense(self.num_classes, activation = 'softmax'))

            return model



