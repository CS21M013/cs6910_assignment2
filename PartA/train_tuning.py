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
from CNN_model import*
from get_data import*
from paths import*

# input image dimensions that we are using for the following task
img_size = (128,128)

'''
Sweep config for the hyperparameter tuning
'''
sweep_config = {
  "name": "Bayesian Sweep",
  "method": "bayes",
  "metric":{
  "name": "val_accuracy", # optimizing wrt to maximizing the validation accuracy
  "goal": "maximize"
  },
  "parameters": {
        
        "activation":{
            "values": ["relu"]
        },
        "filter_size": {
            "values": [(2,2), (3,3), (4,4)]
        },
        "batch_size": {
            "values": [32, 64]
        },
        "padding": {
            "values": ["same","valid"]
        },
        "data_augmentation": {
            "values": [True, False]
        },
        "optimizer": {
            "values": ["adam", "sgd", "rmsprop"]
        },
        "batch_normalization": {
            "values": [True, False]
        },
        "batchnorm_location": {
            "values": ["After", "Before"]
        },
        "num_filters": {
            "values": [32,64, 128]
        },
        "dense_neurons": {
            "values": [64, 128, 256]
        },   
        "dropout_loc": {
            "values": ["conv","dense","all"]
        },
        "dropout": {
            "values": [None, 0.2,0.3]
        },  
        "gap": {
            "values": [False,True]
        }, 
        "filter_multiplier": {
            "values": ["standard","half","double"]
        },       
    }
}

'''
Initializing the wandb sweeps for tuning
'''
sweep_id = wandb.sweep(sweep_config,project=project_name, entity=entity_name)

'''
Training function for sweeping and hyperparameter tuning
'''
def train():
    #project name for wandb initialization
    project_name = 'CS6910-Assignment2-PartA_final'
    # model save path location
    model_save_path = "./TrainedModel/"
    img_size=(128,128)
    #config default values for sweeping and values with only one value. 
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
            epochs = 7,
            batch_size = 32, 
            data_augmentation = False,
            gap = True,
            img_size = img_size
        ) 

    # Initializing the wandb with project name and entity name
    wandb.init(project = project_name, config = config_defaults,entity=entity_name)
    CONFIG = wandb.config

    # Naming convention fo the wandb runs
    wandb.run.name = "Image_recog" + str(CONFIG.num_hidden_cnn_layer) + "_dn_" + str(CONFIG.dense_neurons) + "_opt_" + CONFIG.optimizer + "_dro_" + str(CONFIG.dropout) + "_bs_"+str(CONFIG.batch_size) + "_fm_" + CONFIG.filter_multiplier + "_bnl_" + CONFIG.batchnorm_location + "_dpl_" + CONFIG.dropout_loc

    with tf.device('/device:GPU:0'): 
        # Create object of the CNN model class       
        imageRecog = cnnModel(CONFIG.img_size, CONFIG, 10)
        # Build the model
        model = imageRecog.build_cnnmodel()
        
        # Printing the model summary
        model.summary()

        # Reading the data
        train_generator,validation_generator = get_data(CONFIG.batch_size,CONFIG.data_augmentation,train_data_path)

        # Model compilation
        model.compile(
        optimizer=CONFIG.optimizer,  # Optimizer
        # Loss function to minimize (categorial classification for multiclass classification)
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        # Monitoring accuracy
        metrics=['accuracy'],
        )
        # Model fit for training
        history = model.fit(
                        train_generator,
                        steps_per_epoch = train_generator.samples // CONFIG.batch_size,
                        validation_data = validation_generator, 
                        validation_steps = validation_generator.samples // CONFIG.batch_size,
                        epochs = CONFIG.epochs, 
                        callbacks=[WandbCallback()]
                        )

        # Save the models in the follwoing path
        model.save(model_save_path+wandb.run.name)
        wandb.finish()
        # Returning the model and the history
        return model, history
        
wandb.agent(sweep_id, train, count = 30)