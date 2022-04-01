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
from get_test_data import*
from paths import*

img_width,img_height = 128,128
batch_size=32

# Loading the best model
best_model = keras.models.load_model(best_model_path) #Load the best trained model
best_model.summary()

# Load the test data in batches for evaluation
test_generator = get_test_data(batch_size,test_data_path)
            
#Test loss and accuracy on the shuffled test dataset            
history = best_model.evaluate(test_generator)