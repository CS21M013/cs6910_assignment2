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
# image sizes
img_width,img_height = 128,128
# Batch size
batch_size=32
# Load the test data
test_generator = get_test_data(batch_size,test_data_path,shuffle = False)

# Loading the best model
source_model = keras.models.load_model(best_model_path) #Load the best trained model
# prediction labels
pred_labels = source_model.predict(test_generator)
pred_labels_num = np.argmax(pred_labels, axis = 1)
# Confusion matrix
cm = metrics.confusion_matrix(test_generator.classes, np.argmax(pred_labels, axis = 1))
# plot the figure
plt.figure(figsize=(20,10)) 
# using heatmap for the confusion matrix
sns.heatmap(cm, annot=True)
plt.savefig(save_plots+"Confusion_matrix")
# Display the confusion matrix
plt.show()