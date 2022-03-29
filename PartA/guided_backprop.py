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

#Default image size:
IMG_SIZE = (128, 128)
DATAPATH = test_data_path
MODELPATH = best_model_path
convo_layer = "conv2d_4"
'''
Function to process the image
'''  
def deprocess_image(img):
    img = img.copy()
    # Zero centering the image (subtracting the mean and dividing by the standard deviation)
    img -= img.mean()
    img /= (img.std() + K.epsilon())
    img *= 0.25

    # clip to [0, 1]
    img += 0.5
    # clip the images between 0 and 1
    img = np.clip(img, 0, 1)

    # convert the image to RGB array
    img *= 255
    if K.image_data_format() == 'channels_first':
        img = img.transpose((1, 2, 0))
    img = np.clip(img, 0, 255).astype('uint8')
    return img
 
#Custom gradient function for guided backpropagation. Consists of guided relu function and its gradient.        
'''
Guided Relu function and its derievative for back propagation
'''
@tf.custom_gradient
def guidedRelu(x):
  def derieve(dy):
    # Gradient and the derievative if the Relu function
    return tf.cast(dy>0,"float32") * tf.cast(x>0, "float32") * dy
  # Return the relu if derievative is not true
  return tf.nn.relu(x), derieve

# Guided backpropagation on multiple images:
#------------------------------------------#

def guided_backpropagation(MODELPATH, num_sample_images = 10):
    
    # loading the best model
    model = tf.keras.models.load_model(MODELPATH)

    # guided backpropagated model
    gb_model = Model(
        inputs = [model.inputs],
        outputs = [model.get_layer(convo_layer).output]
    )
    # if layer as activation and if its is relu activation then do the follwing   
    for layer in model.layers:
        if hasattr(layer, 'activation') and layer.activation==tf.keras.activations.relu:
            layer.activation = guidedRelu

    '''
    plotting the images
    '''
    # Subplot for the images
    fig, ax = plt.subplots(1, num_sample_images, figsize=(1*8, 1))
    # subplot for the gradient visualizations
    fig1, ax1 = plt.subplots(1, num_sample_images, figsize=(1*8, 1))
    sample_imgs = tf.convert_to_tensor(test_generator[0][0][:num_sample_images], dtype=tf.float32)
    sample_img_labels = np.array([np.argmax(test_generator[0][1][i]) for i in range(num_sample_images)])
    
    for i in range(num_sample_images):

        with tf.GradientTape() as tape:
            input_img = tf.expand_dims(sample_imgs[i], 0)
            tape.watch(input_img)
            output = gb_model(input_img)[0]

        # obtain the gradient
        gradients = tape.gradient(output,input_img)[0]

        # Display the image
        ax[i].imshow(sample_imgs[i])
        ax[i].set_xlabel(str(sample_img_labels[i]))
        # Display the gradient visualizations by de processing the image into RGB image for displaying
        ax1[i].imshow(deprocess_image(np.array(gradients)))
        ax1[i].set_xlabel(str(sample_img_labels[i]))

    fig.savefig(save_plots+"guided_images")
    fig1.savefig(save_plots+"guided_backprop")    
    plt.show()

    return model, gb_model

#Just to visualise the backpropagated gradients on multiple sample images:
guided_backpropagation(MODELPATH)