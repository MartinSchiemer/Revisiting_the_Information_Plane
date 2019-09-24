"""
author: Martin Schiemer
Callback function for data recording
"""
import numpy as np
import re
import tensorflow as tf
from tensorflow.keras import backend as K


def find_layer_activations(model):
    """
    find layer activation names and stores them in list
    model: keras model
    returns: list of activation names
    """
    activation_name_list = []
    index = 0
    for layer in model.layers:
        if "activation" in layer.name:
            activation_fn_name = re.search(r"function (.+) at",
                                        str(model.layers[index].activation)).groups(1)[0]
            activation_name_list.append(activation_fn_name)
        elif "re_lu" in layer.name:
            if "leaky" in layer.name:
                activation_name_list.append("leaky")
            elif "p" in layer.name:
                activation_name_list.append("prelu")
            elif "thresh" in layer.name:
                activation_name_list.append("thresholded_relu")
            elif "elu" in layer.name:
                activation_name_list.append("elu")
        if "softmax" in layer.name:
            activation_name_list.append("softmax")
        index+=1
    return activation_name_list


def create_layer_name(layer_name, name_nr):
    """
    creates layer names for storage
    layer_name: layer string with unnecessary information
    name_nr: layer index
    returns: layer name
    """
    if "_" in layer_name:
        layer_name_comb = ("{}".format(re.match(r".+?(?=_)", layer_name, flags=0).group() + " " + str(name_nr)))
    else:
        layer_name_comb = (layer_name + " " + str(name_nr))
    return layer_name_comb


def get_weights(weights_dic, model, epoch):
    """
    callback returns matrix with n arrays where n is the number of features and each
    array has m elements where m is the number of neurons
    records weights for each layer for each epoch and stores in provided dictionary dictionary
    weights_dic: dictionary to save weights
    model: keras model
    epoch: current epoch number
    adapted from: from https://stackoverflow.com/questions/42039548/how-to-check-the-weights-after-every-epoc-in-keras-model
    """
    name_nr = 0
    for layer in model.layers:
        if("dense" in layer.name or "cov" in layer.name or "re_lu" in layer.name or
           "softmax" in layer.name or "elu" in layer.name):
            # 1st element of get_weights are weights, second is bias input
            weights_dic[(epoch, name_nr)] = [layer.get_weights()[0]]
            name_nr += 1
        

def get_output_of_all_layers2(outputs_dic, model, epoch, data_input):
    """
    this gets all outputs and stores them in dictionary, make it callback function
    but slower than below
    outputs_dic: dictionary to store outputs
    model: keras model
    epoch: current epoch
    data_input: input data
    """
    name_nr = 0
    for count, layer in enumerate(model.layers):
        if("activation" in layer.name or "re_lu" in layer.name or "softmax" in layer.name or
           "elu" in layer.name):
            intermediate_layer_model = Model(inputs=model.input,
                                             outputs=model.get_layer(layer.name).output)
            intermediate_output = intermediate_layer_model.predict(data_input)#[0]
            
            activation_fn_name = re.search(r"function (.+) at",
                                        str(model.layers[2].activation)).groups(1)[0]
            layer_name = create_layer_name(model.layers[count-1].name + activation_fn_name, name_nr)
            outputs_dic[(epoch, name_nr)] = [intermediate_output, layer_name]

            name_nr += 1



def get_output_of_all_layers(outputs_dic, model, epoch, data_input):
    """
    callback function to extract all network activations for all epochs and store in dictionary
    using backend function
    outputs_dic: dictionary to store outputs
    model: keras model
    epoch: current epoch
    data_input: input data
    """
    name_list = []
    for layer in model.layers:
        if("conv" in layer.name or "dense" in layer.name or "re_lu" in layer.name or
           "softmax" in layer.name or "elu" in layer.name):
            name_list.append(layer.name)
    get_all_layer_outputs = K.function([model.layers[0].input],
                                [l.output for l in model.layers[0:] if "activation" in l.name or 
                                 "re_lu" in l.name or "softmax" in layer.name or
                                 "elu" in layer.name])

    layer_output = get_all_layer_outputs([data_input[:,:]])
    
    name_nr = 0
    for array in layer_output:
        if "conv" in name_list[name_nr]:
            outputs_dic[(epoch, name_nr)] = [np.reshape(array,
                                                        (array.shape[0],
                                                         array.shape[1]*
                                                         array.shape[2]*
                                                         array.shape[3]))]
        else:
            outputs_dic[(epoch, name_nr)] = [array]
        name_nr += 1
        

def record_test_score(model, x_test, y_test):
    """
    records intermediate test scores
    model: keras model
    x_test: test input data 
    y_test: test output data
    returns: test score
    """
    score = model.evaluate(x_test, y_test, verbose=0)
    return score[1]
    
def record_activations(outputs, model, epoch, data_input, x_test, y_test,
                       batchsize=32, record_all_flag=False, rec_test_flag=False,
                       specific_records=[]):
    """
    extracts and records the layer activations for an epoch and stores them in ooutput object
    outputs: output object
    model: keras model
    epoch: current epoch index
    data_input: input features
    x_test: test input
    y_test: test output
    batchsize: batch size 
    record_all_flag: falg that decides if all epochs are recorded or reduced amount
    rec_test_flag: flag that decides if intermediate test scores are recorded
    specific_records: list of specific epochs that shoould be recorded
    """
    # record activation functions
    if epoch == 1:
        act_list = find_layer_activations(model)
        outputs.act_func = act_list
        outputs.batchsize = batchsize
    
    # only record in certain intervals to make computation more feasible
    if((record_all_flag == True) or (epoch <= 30) or (epoch < 300 and epoch%2 == 0) or
       (epoch < 500 and epoch%10 == 0) or (epoch <= 4000 and epoch%100 == 0) or
       (epoch > 4000 and epoch%200 == 0) or (epoch in specific_records)):
        
        if rec_test_flag == True:
            score = record_test_score(model, x_test, y_test)
            outputs.int_model_score[epoch] = score
        
        get_output_of_all_layers(outputs.dic, model, epoch, data_input) 

    
def record_weights(weights, model, epoch, data_input, x_test, y_test, batchsize=32,
                   record_all_flag=False, rec_test_flag=False, specific_records=[]):
    """
    record weights for each epoch for each layer and stores them in weighs object
    weight: weights object
    model: keras model
    epoch: current epoch index
    data_input: input features
    x_test: test input
    y_test: test output
    batchsize: batch size 
    record_all_flag: falg that decides if all epochs are recorded or reduced amount
    rec_test_flag: flag that decides if intermediate test scores are recorded
    specific_records: list of specific epochs that shoould be recorded
    """
    # record activation functions
    if epoch == 1:
        act_list = find_layer_activations(model)
        weights.act_func = act_list
        weights.batchsize = batchsize
    
    # only record in certain intervals to make computation more feasible
    if((record_all_flag == True) or (epoch <= 30) or (epoch < 300 and epoch%2 == 0) or
       (epoch < 500 and epoch%10 == 0) or (epoch <= 4000 and epoch%100 == 0) or
       (epoch > 4000 and epoch%200 == 0) or (epoch in specific_records)):
        
        if rec_test_flag == True:
            score = record_test_score(model, x_test, y_test)
            weights.int_model_score[epoch] = score
        
        get_weights(weights.dic, model, epoch) 


def record_comb(weights, outputs, model, epoch, data_input, x_test, y_test, batchsize=32, record_all_flag=False, rec_test_flag=False, specific_records=[]):
    """
    record weights and activations for each epoch for each layer and stores them in weights and
    outputs object
    weight: weights object
    model: keras model
    epoch: current epoch index
    data_input: input features
    x_test: test input
    y_test: test output
    batchsize: batch size 
    record_all_flag: falg that decides if all epochs are recorded or reduced amount
    rec_test_flag: flag that decides if intermediate test scores are recorded
    specific_records: list of specific epochs that shoould be recorded
    """
    # record activation functions
    if epoch == 1:
        act_list = find_layer_activations(model)
        weights.act_func = act_list
        outputs.act_func = act_list
        weights.batchsize = batchsize
        outputs.batchsize = batchsize
    
    # only record in certain intervals to make computation more feasible
    if((record_all_flag == True) or (epoch <= 30) or (epoch < 300 and epoch%2 == 0) or
       (epoch < 500 and epoch%10 == 0) or (epoch <= 4000 and epoch%100 == 0) or
       (epoch > 4000 and epoch%200 == 0) or (epoch in specific_records)):
        
        if rec_test_flag == True:
            score = record_test_score(model, x_test, y_test)
            weights.int_model_score[epoch] = score
            outputs.int_model_score[epoch] = score
        
        get_weights(weights.dic, model, epoch)
        get_output_of_all_layers(outputs.dic, model, epoch, data_input)      


# from https://stackoverflow.com/questions/51140950/how-to-obtain-the-gradients-in-keras
# maybe here https://stackoverflow.com/questions/45694344/calculating-gradient-norm-wrt-weights-with-keras
def get_weight_grad(model, inputs, outputs, gradient_dic, epoch):
    """
    Gets gradient of model for given inputs and outputs for all weights
    model: keras model
    inputs: input array matrix
    outputs: output object
    gradient_dic: dictionary to store the gradients
    epoc: current epoch index
    """
    grads = model.optimizer.get_gradients(model.total_loss, model.trainable_weights)
    symb_inputs = (model._feed_inputs + model._feed_targets + model._feed_sample_weights)
    f = K.function(symb_inputs, grads)
    x, y, sample_weight = model._standardize_user_data(inputs, outputs)
    output_grad = f(x + y + sample_weight)
    gradient_dic[(epoch)] = output_grad[0]
    