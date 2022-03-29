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
img_size = (128,128)


#sweep config
sweep_config = {
  "name": "Bayesian Sweep",
  "method": "bayes",
  "metric":{
  "name": "val_accuracy",
  "goal": "maximize"
  },
  "parameters": {
        
        "activation":{
            "values": ["relu", "elu", "selu"]
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

sweep_id = wandb.sweep(sweep_config,project='CS6910-Assignment2-PartA_anotherTry_chandra', entity='cs21m007_cs21m013')


def train():

    #config default values for sweeping  
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
            epochs = 5,
            batch_size = 32, 
            data_augmentation = False,
            gap = True,
            img_size = img_size
        ) 



    #wandb.init( config = config_defaults)
    wandb.init(project = 'CS6910-Assignment2-PartA_anotherTry_chandra', config = config_defaults,entity='cs21m007_cs21m013')
    CONFIG = wandb.config


    wandb.run.name = "Image_recog" + str(CONFIG.num_hidden_cnn_layer) + "_dn_" + str(CONFIG.dense_neurons) + "_opt_" + CONFIG.optimizer + "_dro_" + str(CONFIG.dropout) + "_bs_"+str(CONFIG.batch_size) + "_fm_" + CONFIG.filter_multiplier + "_bnl_" + CONFIG.batchnorm_location + "_dpl_" + CONFIG.dropout_loc

    with tf.device('/device:GPU:0'):        
        imageRecog = cnnModel(CONFIG.img_size, CONFIG, 10)
        model = imageRecog.build_cnnmodel()
        
        model.summary()

        #getting the data
        train_generator,validation_generator = get_data(CONFIG.batch_size,CONFIG.data_augmentation)

        model.compile(
        optimizer=CONFIG.optimizer,  # Optimizer
        # Loss function to minimize
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),#'categorical_crossentropy',
        # List of metrics to monitor
        metrics=['accuracy'],
        )
      
        history = model.fit(
                        train_generator,
                        steps_per_epoch = train_generator.samples // CONFIG.batch_size,
                        validation_data = validation_generator, 
                        validation_steps = validation_generator.samples // CONFIG.batch_size,
                        epochs = CONFIG.epochs, 
                        callbacks=[WandbCallback()]
                        )

        model.save('./TrainedModel/'+wandb.run.name)
        wandb.finish()
        return model, history
        
wandb.agent(sweep_id, train, count = 30)
