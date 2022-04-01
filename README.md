# Overview  
Assignment 2 of CS6910 - Fundamentals of Deep Learning  
**(Assignment 2 : Image classification and object detection using CNNs)**  
The task of the assignment was 3 fold.
  1. Building and training a CNN model from scratch for iNaturalist image data classification.
  2. Fine tune a pretrained model on the iNaturalist dataset.
  3. Use a pretrained Object Detection model for a cool application

**The link to the wandb report containing all the aforemnentioned task is** - 
https://wandb.ai/cs21m007_cs21m013/CS6910-Assignment2-PartA_anotherTry_chandra/reports/Assignment-2-Image-classification-and-object-detection-using-CNNs--VmlldzoxNzY2MTcz  

## Part A - Building and training a CNN model from scratch for classification on the Inaturalist12k dataset.
The simple 5 convolutional layer based CNN model has been implemented in the CNN_model.py file.  
The code snippet is also present in the .ipynb file which can also be executed in colab as well as jupyter notebook with GPU requirements. The model class contains 2 functions that are build_cnnmodel and the cnnBlock. The former builds the entire model depending on the hyperparameters that are selected and the cnnBlock function actually returns a block of Convolutional layer, activation layer and maxpooling layer with batchnormalization if necessary.  
The cnnModel class has been called in the train.py file in the hyperparamter tuning part to build a model with set hyper-parameters and in the best_model_train.py to biuld model with best set of hyper-paramters and retain the model for more number of epochs.
