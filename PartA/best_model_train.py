import os

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
#import keras

from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Input, InputLayer, Flatten, Conv2D, BatchNormalization, MaxPooling2D, Activation , GlobalAveragePooling2D
from tensorflow.keras.models import Sequential,  Model

import wandb
import tensorflow as tf
from tensorflow import keras
from sklearn import metrics
import pandas as pd
import cv2
from wandb.keras import WandbCallback
import pathlib
import seaborn as sns
from CNN_model import*
from get_data import*
from paths import*
'''
Setting the hyparameter of the best configuration below for re-training the model with the best set of hyperparameters.
'''
# image size that the model is trainied on
img_size = (128,128)

# sweep config with best set of hyper parameters only for re-training
sweep_config = {
  "name": "Bayesian Sweep",
  "method": "bayes",
  "metric":{
  "name": "val_accuracy",
  "goal": "maximize"
  },
  "parameters": {
        
        "activation":{
            "values": ["elu"]
        },
        "filter_size": {
            "values": [(3,3)]
        },
        "batch_size": {
            "values": [64]
        },
        "padding": {
            "values": ["valid"]
        },
        "data_augmentation": {
            "values": [False]
        },
        "optimizer": {
            "values": ["adam"]
        },
        "batch_normalization": {
            "values": [True]
        },
        "batchnorm_location": {
            "values": ["After"]
        },
        "num_filters": {
            "values": [64]
        },
        "dense_neurons": {
            "values": [128]
        },   
        "dropout_loc": {
            "values": ["dense"]
        },
        "dropout": {
            "values": [0.3]
        },  
        "gap": {
            "values": [True]
        }, 
        "filter_multiplier": {
            "values": ["double"]
        },
    }
}
# Initializing the wandb sweep for best model re-training
sweep_id = wandb.sweep(sweep_config,project=project_name_best, entity=entity_name)

'''
Function for retraining the best model
'''
def best_train(train_data_path,model_save_path):
    img_size=(128,128)
    config_defaults = dict(
            num_hidden_cnn_layer = 5 ,
            activation = 'relu',
            batch_normalization = True,
            batchnorm_location = "After",
            filter_multiplier = "double" ,
            filter_size = (3,3),
            num_filters  = 32,
            dropout = None,
            dropout_loc = "dense",
            pool_size = (2,2),
            padding = 'same',
            dense_neurons = 128,
            num_classes = 10,
            optimizer = 'adam',
            epochs = 8,
            batch_size = 32, 
            data_augmentation = False,
            gap = True,
            img_size = img_size
        ) 
    # wandb initializations
    wandb.init(project = project_name_best, config = config_defaults,entity=entity_name)
    CONFIG = wandb.config

    # Best model run name
    wandb.run.name = "Best_model_another"

    
    # allocating the GPU for colab
    with tf.device('/device:GPU:0'):  
        # creating object of the CNN model class      
        objDetn = cnnModel(CONFIG.img_size, CONFIG, 10)
        # calling the build CNN model function for building the model
        model = objDetn.build_cnnmodel()
        
        # returning the best model summary
        model.summary()

        # loading the train and the validation data generator
        train_generator,validation_generator = get_data(CONFIG.batch_size,CONFIG.data_augmentation,train_data_path)

        # compiling the model
        model.compile(
        optimizer=CONFIG.optimizer,  # Optimizer
        # Loss function to minimize
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),#'categorical_crossentropy for multiclass classification',
        # List of metrics to monitor
        metrics=['accuracy'],
        )
        
        # Model fitting
        history = model.fit(
                        train_generator,
                        steps_per_epoch = train_generator.samples // CONFIG.batch_size,
                        validation_data = validation_generator, 
                        validation_steps = validation_generator.samples // CONFIG.batch_size,
                        epochs = CONFIG.epochs, 
                        callbacks=[WandbCallback()]
                        )

        # Saving the model at the following path
        model.save(model_save_path+wandb.run.name)
        wandb.finish()
        return model, history
		
		
wandb.agent(sweep_id, best_train(train_data_path,model_save_path), count=1)