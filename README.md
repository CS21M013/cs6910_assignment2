# Overview  
Assignment 2 of CS6910 - Fundamentals of Deep Learning  
**(Assignment 2 : Image classification and object detection using CNNs)**  
The task of the assignment was 3 fold.
  1. Building and training a CNN model from scratch for iNaturalist image data classification.
  2. Fine tune a pretrained model on the iNaturalist dataset.
  3. Use a pretrained Object Detection model for a cool application

**The link to the wandb report containing all the aforemnentioned task is** - 
https://wandb.ai/cs21m007_cs21m013/CS6910-Assignment2-PartA_anotherTry_chandra/reports/Assignment-2-Image-classification-and-object-detection-using-CNNs--VmlldzoxNzY2MTcz?accessToken=6onumj31ka4nyjrdbwlnaiqsb77tkxearlcer9rkmjjyh5ngs61n94wu8l808f0n
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
### Retraining the model with the best set of hyper-parameters
After the hyper parameter sweeping is performed and we obtain the best set of hyper parameters that give the best validation accuracy on the dataset, we set the hyperparamters values to that of the best value and retrain the model for larger number of epochs.  
The best set of hyperparameters in our case turned out to be the following:
```
'''
Setting the hyparameter of the best configuration below for re-training the model with the best set of hyperparameters.
'''
# image size that the model is trainied on
img_size = (128,128)

# sweep config with best set of hyper parameters only for re-training
sweep_config = {
  "name": "Bayesian Sweep",
  "method": "bayes",
  "metric":{
  "name": "val_accuracy",
  "goal": "maximize"
  },
  "parameters": {
        
        "activation":{
            "values": ["elu"]
        },
        "filter_size": {
            "values": [(3,3)]
        },
        "batch_size": {
            "values": [64]
        },
        "padding": {
            "values": ["valid"]
        },
        "data_augmentation": {
            "values": [False]
        },
        "optimizer": {
            "values": ["adam"]
        },
        "batch_normalization": {
            "values": [True]
        },
        "batchnorm_location": {
            "values": ["After"]
        },
        "num_filters": {
            "values": [32]
        },
        "dense_neurons": {
            "values": [128]
        },   
        "dropout_loc": {
            "values": ["all"]
        },
        "dropout": {
            "values": [0.3]
        },  
        "gap": {
            "values": [True]
        }, 
        "filter_multiplier": {
            "values": ["double"]
        },
    }
}
```
The best set of hyper parameters need to be set along with the wandb project name, the model save path location and the larger number of epochs to be tested in the best_model_train.py file.  
The retraining can be performed by running the colab notebook or the follwing command.
```
python3 best_model_train.py
```
### Testing and evaluation of the model on the test data.  
The best model after being saved is reloaded and then it is tested giving us the testing accuracy and the loss.  
It can be performed in the colab notebook or by using the following commmand
```
python3 test__eval.py
```
The aforementioned file uses the get_test_data.py file for loading the test data as the training part used the get_data file for loading the training and validation data.
### Sample Image predictions
The sample image predictions are generated by loading the best model and predicting on individual images.
The Sample image prediction plots can be generated in the colab notebook using GPU or using the following comand.  
```
python3 sampleImage_predictions.py
```
### Visualization of the CNNs
In order to learn how the CNNs learn the following have been scripted to generate feature maps of the convolutional layers and the activation layers. Guided back propagation has also been performed on a sample image for 10 neurons to understand the effect of the input on the activation of the neurons.
  1. ```filter_visualization.py```
  2. ```guided_backprop.py```
Both these scripts can be run from the working directory. The aforementioned visualization task can also be performed for in the colab notebook.

**Note:** The paths of the data and the model saved paths and the paths for saving the genrated plots are mentioned in the paths.py file. Still in some places of the code they might need to be mentioned explicitly such as in the train_tuning.py and the best_model_train.py.....

## Part B -   
## Part C - Using a pre-trained model as it is (YOLO-V3)  
The Part C of the assingmnet required us to use a yolo-v3 pre-trained model as it is in application of your choice.  
The application based on object detection using YOLO-V3 is detecting guns,rifles and fires in real time video, images or directly from webcam as well. The higher objective or goal of this application would be to reduce gun violence in different parts of the world andd prevent small accidents caused by fires by detecting them at early stages, when the fire is not devastating enough.
The pretrained weights can be downloaded from the following link - https://drive.google.com/file/d/1HBbJoY7W0NjpMd_-ILklsIXosoVMrIZx/view?usp=sharing  
