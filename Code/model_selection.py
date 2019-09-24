"""
Author: Martin Schiemer
some sample model configurations
"""

import numpy as np
np.random.seed(1337)
# tensorflow properties
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, InputLayer, Dense, Activation, LeakyReLU, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import normalize
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model

def select_model(index, nr_of_epochs, set_name, x_train_shape, y_train):
    """
    creates model and a network name acording to the dataset
    index: flag that decides which model is taken
    nr_of_epochs: how many epoch are max
    set_name: name of the dataset
    x_train_shape: shape of the input data
    y_train: shape of the output data
    returns: model and architecture name
    """
    x_shape_length = len(x_train_shape)
    if len(y_train.shape) > 1:
        y_shape_length = y_train.shape[1]
    else:
        y_shape_length = 1
    amount_of_classes = len(np.unique(y_train))
    print("amount of classes", amount_of_classes)
    print("Input shape: ", x_train_shape, " length: ", x_shape_length)
    # define model
    # mix network with leading TanH
    if index == 1:
        architecture = set_name + str(nr_of_epochs) + "D10T_D7T_D5R_D4R_D3R_D1S"
        
        model = Sequential()
        # inputlayers are needed to allow backend calculations
        if x_shape_length == 2 :
            model.add(InputLayer((x_train_shape[1],)))
        elif x_shape_length == 3 :
            model.add(InputLayer((x_train_shape[1], x_train_shape[2])))
        elif x_shape_length == 4 :
            model.add(InputLayer((x_train_shape[1], x_train_shape[2],
                                  x_train_shape[3])))

        model.add(Dense(10))
        model.add(Activation("tanh"))

        model.add(Dense(7))
        model.add(Activation("tanh"))

        model.add(Dense(5))
        model.add(Activation("relu"))

        model.add(Dense(4))
        model.add(Activation("relu"))

        model.add(Dense(3))
        model.add(Activation("relu"))

        model.add(Flatten())
        model.add(Dense(y_shape_length))
        model.add(Activation("softmax"))
        
    # mix network with leading ReLU
    if index == 2:
        architecture = set_name + str(nr_of_epochs) + "D10R_D7R_D5T_D4T_D3T_D1S"
        
        model = Sequential()
        #model.add(InputLayer((x_train_shape,)))
        if x_shape_length == 2 :
            model.add(InputLayer((x_train_shape[1],)))
        elif x_shape_length == 3 :
            model.add(InputLayer((x_train_shape[1], x_train_shape[2])))
        elif x_shape_length == 4 :
            model.add(InputLayer((x_train_shape[1], x_train_shape[2],
                                  x_train_shape[3])))
            model.add(Flatten())

        model.add(Dense(10))
        model.add(Activation("relu"))

        model.add(Dense(7))
        model.add(Activation("relu"))

        model.add(Dense(5))
        model.add(Activation("tanh"))

        model.add(Dense(4))
        model.add(Activation("tanh"))

        model.add(Dense(3))
        model.add(Activation("tanh"))

        model.add(Flatten())
        model.add(Dense(y_shape_length))
        model.add(Activation("softmax"))
        
    # ReLU network    
    if index == 3:
        architecture = set_name + str(nr_of_epochs) + "D10R_D7R_D5R_D4R_D3R_D1S"
        
        model = Sequential()
        if x_shape_length == 2 :
            model.add(InputLayer((x_train_shape[1],)))
        elif x_shape_length == 3 :
            model.add(InputLayer((x_train_shape[1], x_train_shape[2])))
        elif x_shape_length == 4 :
            model.add(InputLayer((x_train_shape[1], x_train_shape[2],
                                  x_train_shape[3])))

        model.add(Dense(10))
        model.add(Activation("relu"))

        model.add(Dense(7))
        model.add(Activation("relu"))

        model.add(Dense(5))
        model.add(Activation("relu"))

        model.add(Dense(4))
        model.add(Activation("relu"))

        model.add(Dense(3))
        model.add(Activation("relu"))

        model.add(Flatten())
        model.add(Dense(y_shape_length))
        model.add(Activation("softmax"))
        
    # TanH network
    if index == 4:
        architecture = set_name + str(nr_of_epochs) + "D10T_D7T_D5T_D4T_D3T_D1S"
        
        model = Sequential()
        if x_shape_length == 2 :
            model.add(InputLayer((x_train_shape[1],)))
        elif x_shape_length == 3 :
            model.add(InputLayer((x_train_shape[1], x_train_shape[2])))
        elif x_shape_length == 4 :
            model.add(InputLayer((x_train_shape[1], x_train_shape[2],
                                  x_train_shape[3])))
            model.add(Flatten())

        model.add(Dense(10))
        model.add(Activation("tanh"))

        model.add(Dense(7))
        model.add(Activation("tanh"))

        model.add(Dense(5))
        model.add(Activation("tanh"))

        model.add(Dense(4))
        model.add(Activation("tanh"))

        model.add(Dense(3))
        model.add(Activation("tanh"))

        model.add(Flatten())
        model.add(Dense(y_shape_length))
        model.add(Activation("softmax"))
        
    # convolutional network with ReLU
    if index == 5:
        architecture = set_name + str(nr_of_epochs) + "COV32R_COV64R_D10R_D5R_D" + str(amount_of_classes) + "Soft"
        model = Sequential()
        if x_shape_length == 2 :
            model.add(InputLayer((x_train_shape[1])))
        elif x_shape_length == 3 :
            model.add(InputLayer((x_train_shape[1], x_train_shape[2])))
        elif x_shape_length == 4 :
            model.add(InputLayer((x_train_shape[1],
                                  x_train_shape[2], x_train_shape[3])))

        model.add(Conv2D(16, kernel_size=(3,3), strides=(1,1),input_shape=(x_train_shape[0],
                                                                           x_train_shape[1],
                                                                           x_train_shape[2])))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        
        model.add(Conv2D(32, kernel_size=(5, 5), strides=(1,1)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        model.add(Flatten())
        model.add(Dense(7))
        model.add(Activation("relu"))
        
        model.add(Dense(5))
        model.add(Activation("relu"))

        model.add(Flatten())
        model.add(Dense(y_shape_length))
        model.add(Activation("softmax"))
    
    # convolutional network with TanH
    if index == 6:
        architecture = set_name + str(nr_of_epochs) + "COV32T_COV64T_D10T_D5T_D" + str(amount_of_classes) +"Soft"
        model = Sequential()
        if x_shape_length == 2 :
            model.add(InputLayer((x_train_shape[1])))
        elif x_shape_length == 3 :
            model.add(InputLayer((x_train_shape[1], x_train_shape[2])))
        elif x_shape_length == 4 :
            model.add(InputLayer((x_train_shape[1],
                                  x_train_shape[2], x_train_shape[3])))

        model.add(Conv2D(16, kernel_size=(3,3), strides=(1,1),
                         input_shape=(x_train_shape[0],x_train_shape[1],x_train_shape[2])))
        model.add(Activation("tanh"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        
        model.add(Conv2D(32, kernel_size=(5, 5), strides=(1,1)))
        model.add(Activation("tanh"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        model.add(Flatten())
        model.add(Dense(7))
        model.add(Activation("tanh"))
        
        model.add(Dense(5))
        model.add(Activation("tanh"))

        model.add(Flatten())
        model.add(Dense(y_shape_length))
        model.add(Activation("softmax"))  
        
    # Same layer size network ReLU
    if index == 7:
        architecture = set_name + str(nr_of_epochs) + "D5R_D5R_D5R_D5R_D5R_D1S"
        
        model = Sequential()
        if x_shape_length == 2 :
            model.add(InputLayer((x_train_shape[1],)))
        elif x_shape_length == 3 :
            model.add(InputLayer((x_train_shape[1], x_train_shape[2])))
        elif x_shape_length == 4 :
            model.add(InputLayer((x_train_shape[1], x_train_shape[2],
                                  x_train_shape[3])))

        model.add(Dense(5))
        model.add(Activation("relu"))

        model.add(Dense(5))
        model.add(Activation("relu"))

        model.add(Dense(5))
        model.add(Activation("relu"))

        model.add(Dense(5))
        model.add(Activation("relu"))

        model.add(Dense(5))
        model.add(Activation("relu"))

        model.add(Flatten())
        model.add(Dense(y_shape_length))
        model.add(Activation("softmax"))
    
    # Same layer size network TanH
    if index == 8:
        architecture = set_name + str(nr_of_epochs) + "D5T_D5T_D5T_D5T_D5T_D1S"
        
        model = Sequential()
        if x_shape_length == 2 :
            model.add(InputLayer((x_train_shape[1],)))
        elif x_shape_length == 3 :
            model.add(InputLayer((x_train_shape[1], x_train_shape[2])))
        elif x_shape_length == 4 :
            model.add(InputLayer((x_train_shape[1], x_train_shape[2],
                                  x_train_shape[3])))

        model.add(Dense(5))
        model.add(Activation("tanh"))

        model.add(Dense(5))
        model.add(Activation("tanh"))

        model.add(Dense(5))
        model.add(Activation("tanh"))

        model.add(Dense(5))
        model.add(Activation("tanh"))

        model.add(Dense(5))
        model.add(Activation("tanh"))

        model.add(Flatten())
        model.add(Dense(y_shape_length))
        model.add(Activation("softmax"))
         
    # bottleneck network with TanH    
    if index == 9:
        architecture = set_name + str(nr_of_epochs) + "D12T_D3T_D2T_D12T_D2T_D1S"
        
        model = Sequential()
        if x_shape_length == 2 :
            model.add(InputLayer((x_train_shape[1],)))
        elif x_shape_length == 3 :
            model.add(InputLayer((x_train_shape[1], x_train_shape[2])))
        elif x_shape_length == 4 :
            model.add(InputLayer((x_train_shape[1], x_train_shape[2],
                                  x_train_shape[3])))

        model.add(Dense(12))
        model.add(Activation("tanh"))

        model.add(Dense(3))
        model.add(Activation("tanh"))

        model.add(Dense(2))
        model.add(Activation("tanh"))

        model.add(Dense(12))
        model.add(Activation("tanh"))

        model.add(Dense(2))
        model.add(Activation("tanh"))

        model.add(Flatten())
        model.add(Dense(y_shape_length))
        model.add(Activation("softmax"))
        
    # bottleneck network with ReLU
    if index == 10:
        architecture = set_name + str(nr_of_epochs) + "D12R_D3R_D2R_D12R_D2R_D1S"
        
        model = Sequential()
        if x_shape_length == 2 :
            model.add(InputLayer((x_train_shape[1],)))
        elif x_shape_length == 3 :
            model.add(InputLayer((x_train_shape[1], x_train_shape[2])))
        elif x_shape_length == 4 :
            model.add(InputLayer((x_train_shape[1], x_train_shape[2],
                                  x_train_shape[3])))

        model.add(Dense(12))
        model.add(Activation("relu"))

        model.add(Dense(3))
        model.add(Activation("relu"))

        model.add(Dense(2))
        model.add(Activation("relu"))

        model.add(Dense(12))
        model.add(Activation("relu"))

        model.add(Dense(2))
        model.add(Activation("relu"))

        model.add(Flatten())
        model.add(Dense(y_shape_length))
        model.add(Activation("softmax"))
    
    # leaky ReLU network
    if index == 11:
        architecture = set_name + str(nr_of_epochs) + "D10LR_D7LR_D5LR_D4LR_D3LR_D1S"
        
        model = Sequential()
        #model.add(InputLayer((x_train_shape,)))
        if x_shape_length == 2 :
            model.add(InputLayer((x_train_shape[1],)))
        elif x_shape_length == 3 :
            model.add(InputLayer((x_train_shape[1], x_train_shape[2])))
        elif x_shape_length == 4 :
            model.add(InputLayer((x_train_shape[1], x_train_shape[2],
                                  x_train_shape[3])))
            model.add(Flatten())

        model.add(Dense(10))
        model.add(LeakyReLU(alpha=0.1))

        model.add(Dense(7))
        model.add(LeakyReLU(alpha=0.1))

        model.add(Dense(5))
        model.add(LeakyReLU(alpha=0.1))

        model.add(Dense(4))
        model.add(LeakyReLU(alpha=0.1))

        model.add(Dense(3))
        model.add(LeakyReLU(alpha=0.1))

        model.add(Flatten())
        model.add(Dense(y_shape_length))
        model.add(Activation("softmax"))
        
    return model, architecture