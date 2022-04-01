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
### Building the simple CNN model
The simple 5 convolutional layer based CNN model has been implemented in the CNN_model.py file.  
The code snippet is also present in the .ipynb file which can also be executed in colab as well as jupyter notebook with GPU requirements. The model class contains 2 functions that are build_cnnmodel and the cnnBlock. The former builds the entire model depending on the hyperparameters that are selected and the cnnBlock function actually returns a block of Convolutional layer, activation layer and maxpooling layer with batchnormalization if necessary.  
The cnnModel class has been called in the train.py file in the hyperparamter tuning part to build a model with set hyper-parameters and in the best_model_train.py to biuld model with best set of hyper-paramters and retain the model for more number of epochs.

The package dependencies are provided in the libraries_required.txt file in the Part A folder and the libraries can be easily installed using:  
```
pip install -r libraries_required.txt
```
The model can be built by using the following command even though it is not necessary to run separately
```
python3 CNN_model.py
```
### Training and hyperparameter tuning using Wandb Sweeps  
Wandb sweeps are used to track the validation accuracy and loss including the training acuracy and loss of the model. Sweeps are run for several hyper-parameter configurations (hyperparameter tuning or optimization). The Bayesian method was used to get better results in much less number of sweeps.  
The set of hyper parameter values used for the tuning part is as follows.
```
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
```
The hyper-parameter tuning can be performed after setting the appropriate parameter values. The correct project name of the wandb initialization and the model save path name needs to be specified in the train() function for saving the corresponding models for future uses.
The colab notebook can be used for execution or the following comand may be used:
```
python3 train_tuning.py
```
