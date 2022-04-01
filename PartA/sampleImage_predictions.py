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
from get_test_data import*

#To let the gpu memory utilisiation grow as per requirement
physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
#Invalid device or cannot modify virtual devices once initialized.
    pass

# The dimensions of our input image
img_width,img_height = 128, 128
# Batch size
batch_size=32
wandb.init(project = "partA_plots", entity='cs21m007_cs21m013')
# Load the test data
test_generator = get_test_data(batch_size,test_data_path,shuffle=False)

#Model trained from scratch
source_model = keras.models.load_model(best_model_path) #Load the best trained model


# Sample image predictions
ROWS = 3
COLUMNS = 10 
ix = 1 
fig,ax = plt.subplots(ROWS,COLUMNS,figsize=(6*ROWS,6))
for i in range(ROWS): 
    for j in range(COLUMNS): 
        # Creating the subplots for the sample images
        idx = np.random.choice(len(test_generator[4*j][0])) 
        img = test_generator[4*j][0][idx] 
        ax = plt.subplot(ROWS, COLUMNS, ix) 

        ax.set_xticks([]) 
        ax.set_yticks([])
        # Creating the images of the subplots 
        plt.imshow(img) 
        plt.xlabel(
                    "True: " + str(np.argmax(test_generator[4*j][1][idx])) +"\n" + "Pred: " + 
                    str(np.argmax(source_model.predict(img.reshape(1,128,128,3))))
                   )     
        ix += 1 
# Save the images
plt.savefig(save_plots+"sample_prediction")
wandb.log({"sample_prediction": plt})
plt.show()