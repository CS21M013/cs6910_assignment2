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


#To let the gpu memory utilisiation grow as per requirement
physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
#Invalid device or cannot modify virtual devices once initialized.
    pass

'''
plotting the subplots for the filter visualizations
'''
def plot_maps(feature_maps,ROWS,COLUMNS,savePath):
		ROWS = ROWS
		COLUMNS = COLUMNS 
		ix = 1
		for _ in range(ROWS):
			for _ in range(COLUMNS):
				# specify subplot and turn of axis
				ax = plt.subplot(ROWS, COLUMNS, ix)
				ax.set_xticks([])
				ax.set_yticks([])
				# plot filter channel in grayscale
				plt.imshow(feature_maps[0, :, :, ix-1])
				ix += 1
		# show the figure
		plt.tight_layout()
		plt.savefig(savePath)
		plt.show()

'''
Function to return the class names
'''
def class_names(DATAPATH):
    # Return the list of the class names
		class_name=[]
		for dir1 in np.sort(os.listdir(DATAPATH)):
				class_name.append(dir1)
		return class_name



class_names = class_names(test_data_path)
target_dict = {k: v for v, k in enumerate(np.unique(class_names))} 
class_label_names_dict = {str(k): v for k, v in enumerate(np.unique(class_names))} 
# The dimensions of our input image

img_width,img_height = 128, 128
batch_size=32

test_generator = get_test_data(batch_size,test_data_path,shuffle=False)

# Loading the best model
source_model = keras.models.load_model(best_model_path) #Load the best trained model

#1st Convolutional layer:
layer_name = "conv2d"  # convolutional layer
activation_layer_name = "activation" # Activation layer

# get the layer
layer = source_model.get_layer(name=layer_name)
# get the activation layer
activation_layer = source_model.get_layer(name=activation_layer_name)

# Extract features from the 1st convo-layer
feature_extractor = keras.Model(inputs=source_model.inputs, outputs=layer.output)
# Extract features from the 1st activation layer
feature_extractor_activation = keras.Model(inputs=source_model.inputs, outputs=activation_layer.output)

# get the filters and the biases (weights)
filters, biases = layer.get_weights()


# normalize filter values to 0-1 so we can visualize them
filter_min, filter_max = filters.min(), filters.max()
filters = (filters - filter_min) / (filter_max - filter_min)

# Create a random batch of images
batch = np.random.choice(int(2000/32))
img_index = np.random.choice(32)    
# get a random image
img = test_generator[batch][0][img_index]
img = np.expand_dims(img, axis = 0)
# Get the label of the random image
img_true_label = test_generator[batch][1][img_index]

'''plot the test image:'''
plt.figure()
plt.xlabel(class_label_names_dict[str(np.argmax(img_true_label))])
plt.imshow(img[0])
plt.title("Image")
plt.savefig(save_plots+"Image")
plt.show()



#Extract the feature maps and feature activation maps
feature_maps = feature_extractor(img) 
feature_maps_activation = feature_extractor_activation(img) 

#64 filters in the first layer of the Best model:
ROWS = 8
COLUMNS = 8
plot_maps(feature_maps,ROWS,COLUMNS,save_plots+"Feature_map")

ROWS = 8
COLUMNS = 8
plot_maps(feature_maps_activation,ROWS,COLUMNS,save_plots+"activation_maps")