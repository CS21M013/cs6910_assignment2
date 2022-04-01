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

plt.figure(figsize=(4, 6))
num_sample_images=10
sample_imgs = tf.convert_to_tensor(test_generator[0][0][:num_sample_images], dtype=tf.float32)
sample_img_labels = np.array([np.argmax(test_generator[0][1][i]) for i in range(num_sample_images)])
wandb.init(project = "partA_plots", entity='cs21m007_cs21m013')
plt.savefig(save_plots+"guided_image")
plt.imshow(sample_imgs[0]) # Displaying the image that we are performinng guided back prop on.
wandb.log({"Guided_test_image": plt})
plt.show()

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

# Guided backpropagation on a single image for multiple neurons:
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
    sample_imgs = tf.convert_to_tensor(test_generator[0][0][:num_sample_images], dtype=tf.float32)
    sample_img_labels = np.array([np.argmax(test_generator[0][1][i]) for i in range(num_sample_images)])
    conv_output_shape = model.get_layer(convo_layer).output.shape[1:]
    #plt.imshow(sample_imgs[0])
    plt.figure(figsize=(20, 10))
    #plt.set_label(str(sample_img_labels[0]))
    for i in range(10):
        node_x = np.random.randint(0, conv_output_shape[0])

        node_y = np.random.randint(0, conv_output_shape[1])

        node_z = np.random.randint(0, conv_output_shape[2])

        # Focus on single neuron of the conv layer (mask for this)
        mask = np.zeros((1, *conv_output_shape), dtype="float")
        mask[0, node_x, node_y, node_z] = 1
        with tf.GradientTape() as tape:
            inputs = tf.cast(np.array([np.array(sample_imgs[0])]), tf.float32)
            tape.watch(inputs)
            outputs = gb_model(inputs) * mask

        # obtain the gradient
        gradients = tape.gradient(outputs,inputs)[0]
        img_gb = np.dstack((gradients[:, :, 0], gradients[:, :, 1], gradients[:, :, 2],)) 
        plt.subplot(2, 5, i+1)
        plt.imshow(deprocess_image(img_gb))
    plt.tight_layout() 
    plt.savefig(save_plots+"guided_backprop")
    plt.show()

    return model, gb_model

#Just to visualise the backpropagated gradients on multiple sample images:
guided_backpropagation(MODELPATH)