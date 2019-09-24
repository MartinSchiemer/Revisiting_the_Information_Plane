"""
Author: Martin Schiemer
Collects training and testing data from various sources and transforms them in the required format
"""

import numpy as np
np.random.seed(1337)
import scipy.io as sio
import os
import sys
import random
random.seed(1337)
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
# splitting in training and testing
from sklearn.model_selection import train_test_split
# dataset imports
from tensorflow.keras.datasets import mnist
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.datasets import boston_housing


def shuffle_in_unison_inplace(a, b):
    """
    Shuffle the arrays randomly
    a: array to shuffle in the same way as b
    b: array to shuffle in the same way as a
    taken from:
    https://github.com/artemyk/ibsgd/blob/master/utils.py
    """
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def load_data(filename, random_labels=False):
    """
    Load the data
    filename: the name of the dataset
    random_labels: True if we want to return random labels to the dataset
    return object with data and labels
    returns: Opening the Blackbox dataset
    taken and adapted from:
    https://github.com/ravidziv/IDNNs
    """
    C = type('type_C', (object,), {})
    data_sets = C()
    d = sio.loadmat("../Data/"+ filename + '.mat')
    F = d['F']
    y = d['y']
    C = type('type_C', (object,), {})
    data_sets = C()
    data_sets.data = F
    data_sets.labels = np.squeeze(np.concatenate((y[None, :], 1 - y[None, :]), axis=0).T)
    # If we want to assign random labels to the  data
    if random_labels:
        labels = np.zeros(data_sets.labels.shape)
        labels_index = np.random.randint(low=0, high=labels.shape[1], size=labels.shape[0])
        labels[np.arange(len(labels)), labels_index] = 1
        data_sets.labels = labels
    return data_sets


def trans_MNIST(name, X_train, y_train, X_test, y_test):
    """
    transforms mnist to either image format or continous feature array and normalizes
    after that transforms output data to categorical array
    name: name of the dataset
    X_train: training input
    y_train: training output
    X_test: test input
    y_test: test output
    returns: transformed training and test input/output
    """
    #transform data for cnn or dense layers
    if "cov" in name:
        X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
        X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
    else:
        X_train = X_train.reshape(X_train.shape[0], 784)
        X_test = X_test.reshape(X_test.shape[0], 784)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    # normalize data
    X_train = X_train/255
    X_test = X_test/255
    # transform to one hot encoding
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    
    return X_train, X_test, y_train, y_test 

# extract an equal amount of random indices for specified numbers for MNIST
def MNIST_extract_shuffled_indices(y_train, list_of_nrs, samples_amount=-1):
    """
    selects indices for mnist traning data from input and output and shuffles them
    returns: shuffled indices
    """
    index_list = []
    for number in list_of_nrs:
        indices = np.where(y_train == number)
        if samples_amount == -1:
            
            index_list.append(indices[0][:indices[0].size])
            #index_list.append(indices[0][:indices[0].size])
        else:
            #seed for reproducibility
            random.seed(1337)
            random.shuffle(indices[0])
            index_list.append(indices[0][:samples_amount])
    indices = np.array(index_list).flatten()
    #random.seed(1337)
    #random.shuffle(indices)
    return indices

def retrieve_MNIST(name, shuffle, samples_per_class, list_of_nrs):
    """
    retireves mnist dataset with the required data in the required format
    name: name of dataset
    shuffle: flag if shuffle or not
    samples_per_class: how many samples per class from mnist
    list_of_nrs: which number classes
    returns: transformed mnist data
    """
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
        
    # extract indices of the wanted numbers
    train_filter = np.isin(y_train, list_of_nrs)
    test_filter = np.isin(y_test, list_of_nrs)
    # extract wanted numbers
    X_train, y_train = X_train[train_filter], y_train[train_filter]
    X_test, y_test = X_test[test_filter], y_test[test_filter]
    
    #if samples per class is specified find random indices
    if samples_per_class != -1:
        # extract indices of amount of samples per class wanted
        indices = MNIST_extract_shuffled_indices(y_train, list_of_nrs, samples_per_class)
        # apply indices on train data
        X_train, y_train = X_train[indices], y_train[indices]
    if shuffle == True:
         X_train, y_train = shuffle_in_unison_inplace(X_train, y_train)

    print("X_train shape ", X_train.shape)
    print("y_train shape ", y_train.shape)
    return trans_MNIST(name, X_train, y_train, X_test, y_test)

def retrieve_FMNIST(name, shuffle, samples_per_class, list_of_nrs):
    """
    retireves fashion-mnist dataset with the required data in the required format
    name: name of dataset
    shuffle: flag if shuffle or not
    samples_per_class: how many samples per class from mnist
    list_of_nrs: which number classes
    returns: transformed fashion mnist data
    """
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    
    # extract indices of the wanted numbers
    train_filter = np.isin(y_train, list_of_nrs)
    test_filter = np.isin(y_test, list_of_nrs)
    # extract wanted numbers
    X_train, y_train = X_train[train_filter], y_train[train_filter]
    X_test, y_test = X_test[test_filter], y_test[test_filter]
    
    #if samples per class is specified find random indices
    if samples_per_class != -1:
        # extract indices of amount of samples per class wanted
        indices = MNIST_extract_shuffled_indices(y_train, list_of_nrs, samples_per_class)
        # apply indices on train data
        X_train, y_train = X_train[indices], y_train[indices]
    
    return trans_MNIST(name, X_train, y_train, X_test, y_test)

def retrieve_Tishby(shuffle):
    """
    retireves opening the balckbox dataset with the required data in the required format
    shuffle: flag if shuffle or not
    returns: training and testing data
    """
    name = "var_u"
    data = load_data(name)
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.labels,
                                                            test_size=0.2, random_state=42)
    if shuffle == True:
        X_train, y_train = shuffle_in_unison_inplace(X_train, y_train)
    
    return X_train, X_test, y_train, y_test


# shuffle set to false, since the original paper of Schwartz-Ziv and Tishby does not shuffle
def select_data(set_name, shuffle=False, samples_per_class = -1, list_of_nrs=[0,1,2,3,4,5,6,7,8,9]):
    """
    starts data collection process depending on given set name
    name: name of dataset
    shuffle: flag if shuffle or not
    samples_per_class: how many samples per class from mnist
    list_of_nrs: which number classes
    returns: dataset
    """
    if samples_per_class < -1:
        samples_per_class = -1
    
    print ("Loading " + set_name + " Data...")
    if "fmnistcov" in set_name:
        return retrieve_FMNIST(set_name, shuffle, samples_per_class, list_of_nrs)
    elif "mnist" in set_name:
        return retrieve_MNIST(set_name, shuffle, samples_per_class, list_of_nrs)
    elif "tishby" in set_name:
        return retrieve_Tishby(shuffle)

