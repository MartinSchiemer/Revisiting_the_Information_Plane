"""
Author: Martin Schiemer
Performs entropy and mutual information calculations and returns object holding parameters
Mostly following the implementation of https://github.com/ravidziv/IDNNs
"""

import numpy as np
np.random.seed(1337)
import math
import multiprocessing
from joblib import Parallel, delayed

NUM_CORES = multiprocessing.cpu_count()

def find_max_min_weight(weights):
    """
    finds the maximum and minimum weights and stores them in the weight object
    weights: weight object
    """
    max_w = -float("inf")
    min_w = float("inf")
    for entry in weights.dic:
        # if theres no entry yet initialize to infinity to accomodate
        # for all possible number spaces
        if(entry[1] not in weights.max):
            weights.max[entry[1]] = max_w
        if(entry[1] not in weights.min):
            weights.min[entry[1]] = min_w
        # compare max and min for each layer for each iteration
        if(np.amax(weights.dic[entry][0]) > weights.max[entry[1]]):
            weights.max[entry[1]] = np.amax(weights.dic[entry][0])
        if(np.amin(weights.dic[entry][0]) < weights.min[entry[1]]):
            weights.min[entry[1]] = np.amin(weights.dic[entry][0])

            
# finds the min and max of the activations for each layer to find bounds
def find_max_min_activation(outputs_dic):
    """
    finds the maximum and minimum activations for each layer and epoch and stores them in
    dictionaries 
    outputs_dic: dictionary with activations
    returns: max and min dictionaries
    """
    mx = {}
    mn = {}
    # intialize min/max with infinity so any value is smaller/bigger
    max_o = -float("inf")
    min_o = float("inf")
    
    for entry in outputs_dic:
        # if theres no entry yet initialize to infinity to accomodate
        # for all possible number spaces
        if(entry[1] not in mx):
            mx[entry[1]] = max_o
        if(entry[1] not in mn):
            mn[entry[1]] = min_o
        # compare max and min for each layer for each iteration
        if(np.amax(outputs_dic[entry]) > mx[entry[1]]):
            mx[entry[1]] = np.amax(outputs_dic[entry])
        if(np.amin(outputs_dic[entry]) < mn[entry[1]]):
            mn[entry[1]] = np.amin(outputs_dic[entry])
    return mx, mn


def extract_unique(obj):
    """
    turns features into continous byte array to squeeze input in same shape as labels
    then extracts unique values
    obj: array matrix of values
    returns: inverse of unique arrays, unique counts
    taken from:
    https://github.com/ravidziv/IDNNs/blob/master/idnns/information/information_process.py
    """
    if len(obj.shape) > 1:
        b = np.ascontiguousarray(obj).view(np.dtype((np.void, obj.dtype.itemsize * obj.shape[1])))
    else:
        b = np.ascontiguousarray(obj).view(np.dtype((np.void, obj.dtype.itemsize * 1)))
    unique_values, unique_indices, unique_inverse, unique_counts = \
                                    np.unique(b, return_index=True,
                                              return_inverse=True, return_counts=True)
    
    return unique_inverse, unique_counts 
            
            
def extract_inout_probs(features, output):
    """
    calculate input and output probabilities
    features: array matrix
    output: array matrix
    returns: unique inverses of arrays, probabilities of continous arrays
    mostly taken from:
    https://github.com/ravidziv/IDNNs/blob/master/idnns/information/information_process.py
    """
    
    unique_inverse_x, unique_counts_x = extract_unique(features)
    unique_inverse_y, unique_counts_y = extract_unique(output)
    
    pxs = unique_counts_x / float(np.sum(unique_counts_x))
    pys = unique_counts_y / float(np.sum(unique_counts_y))
    
    return unique_inverse_x, unique_inverse_y, pxs, pys



def calc_bins(name, bin_size_or_nr, bin_amount, max_v, min_v):
    """
    creates bins in accordance to the known bounds of the activation functions
    name: name of the activation function
    bin_size_or_nr: flag that decides whether bin size or amount is used
    bin_amount: amount of bins OR size of bins
    max_v: maximum value of activation
    min_v: minimum value of activation
    returns: edges of bins, amount of bins 
    """
    # create bins in accordance to the known bounds
    if ("tanh" in name):
        # if true fit bin size as often as possible in distance betweem min and max activation
        if bin_size_or_nr == True:
            bin_amount = np.floor((1 - (-1)) / bin_amount).astype("int")
        # + 1 since the we need one more edge to have the wanted bin amount
        bin_edges = np.linspace(-1, 1, bin_amount + 1)
    elif ("sigmoid" in name or "softmax" in name):
        if bin_size_or_nr == True:
            bin_amount = np.floor((1 - (0)) / bin_amount).astype("int")
        bin_edges = np.linspace(0, 1, bin_amount + 1)
    elif ("prelu" in name or "leaky" in name):
        if bin_size_or_nr == True:
            bin_amount = np.floor((max_v - (min_v)) / bin_amount).astype("int")
        bin_edges = np.linspace(min_v, max_v, bin_amount + 1)
    elif ("linear" in name or "relu" in name or "softplus" in name):
        # 0 is lower bound max assumed as upper bound to allow binning
        if bin_size_or_nr == True:
            bin_amount = np.floor((max_v - (0)) / bin_amount).astype("int")
        bin_edges = np.linspace(0, max_v, bin_amount + 1)
    elif ("weight" in name):
        if bin_size_or_nr == True:
                bin_amount = np.floor((max_v-(min_v)) / bin_amount).astype("int")
        bin_edges = np.linspace(min_v, max_v, bin_amount + 1)
    else:
        if bin_size_or_nr == True:
            bin_amount = np.floor((max_v - (min_v)) / bin_amount).astype("int")
        bin_edges = np.linspace(min_v, max_v, bin_amount + 1)
    #allows max value to fall into same bin and not create new bin
    bin_edges[-1] = bin_edges[-1] + 0.0001
    
    return bin_edges, bin_amount
      
# calculates the probabilities of the activation functions for each layer and epoch
def calc_act_probs(par_obj, bin_size_or_nr, bin_amount, activation_name_list):
    """
    calculates the activation probabilities and records empty bins
    
    par_obj: parameter object e.g. output object
    bin_size_or_nr: flag that decides whether bin size or amount is used
    bin_amount: amount OR size of bins
    activation_name_list: list of the activation names
    returns: activation probabilities, binned array, amount of bins, dictionary with unused bins,
    dictionary with the bin edges
    """
    p_act = {}
    dig = {}
    unused_bins_dic = {}
    bin_edge_dic = {}
    bin_am = {}
    
    for key in par_obj.dic.keys():
        name = activation_name_list[key[1]]
        
        bin_edges, bin_am[key] = calc_bins(name, bin_size_or_nr, bin_amount,
                              par_obj.max[key[1]], par_obj.min[key[1]])
        
        bin_edge_dic[key] = bin_edges
        
        # calculate bin probabilities. Taken and adapted from:
        #https://github.com/ravidziv/IDNNs/blob/master/idnns/information/information_process.py    
        digitized = bin_edges[np.digitize(np.squeeze(par_obj.dic[key][0].reshape(1, -1)),
                                          bin_edges) - 1].reshape(len(par_obj.dic[key][0]), -1)

        # record unused bins
        unused_bins = [bin_edge for bin_edge in bin_edges if
                       np.count_nonzero(digitized == bin_edge) <= 1]
        unused_bins_dic[key] = len(unused_bins)
        
        # extract uniques
        unique_inverse_t, unique_counts = extract_unique(digitized)
        
        # calc activation probability
        p_ts = unique_counts / float(sum(unique_counts))

        
        p_act[key] = [p_ts, name]
        dig[key] = [digitized, name]
    
    return p_act, dig, bin_am, unused_bins_dic, bin_edge_dic
        
        
def calc_w_probs(par_obj, bin_size_or_nr, bin_amount, activation_name_list):
    """
    calculates weight probabilitites
    par_obj: parameter object e.g. output object
    bin_size_or_nr: flag that decides whether bin size or amount is used
    bin_amount: amount OR size of bins
    activation_name_list: list of the activation names
    returns: weight probs, binned weights
    """
    p_w = {}
    dig = {}

    for key in par_obj.dic.keys():
        
        name = activation_name_list[key[1]]
        
        bin_edges = calc_bins(name, bin_size_or_nr, bin_amount,
                              par_obj.max[key[1]], par_obj.min[key[1]])
        
        digitized = bin_edges[np.digitize(np.squeeze(par_obj.dic[key][0].reshape(1, -1)),
                                          bin_edges) - 1].reshape(len(par_obj.dic[key][0]), -1)
        
        unique_inverse_t, unique_counts = extract_unique(digitized)
        
        p_ts = unique_counts / float(sum(unique_counts))
        
        p_w[key] = [p_ts, name]
        dig[key] = [digitized, name]
        
    return p_w, dig
        
        
# extracts input, output and activation probabilities
def calc_probs(x, y, prob_obj, par_obj, bin_size_or_nr, bin_amount):
    """
    calculates input, output and activation probs and stores them in probability object
    x: input
    y: output
    prob_obj: object that holds probabilities
    par_obj: parameter object e.g. output object
    bin_size_or_nr: flag that decides whether bin size or amount is used
    bin_amount: amount OR size of bins
    
    """
    
    prob_obj.inverse_x, prob_obj.inverse_y, prob_obj.px, prob_obj.py = \
                                                    extract_inout_probs(x, y)
    
    activation_name_list = par_obj.act_func
    #p_act, dig, bin_am, bin_am, unused_bins_dic, bin_edge_dic
    if("output" in par_obj.name):
        prob_obj.p_layer_act, par_obj.digitized, prob_obj.tot_bins, prob_obj.unused_bins, prob_obj.bin_edges = calc_act_probs(par_obj, bin_size_or_nr, bin_amount, activation_name_list)
            
    elif("weight" in par_obj.name):
        prob_obj.p_layer_w, par_obj.digitized = calc_w_probs(par_obj, bin_size_or_nr, bin_amount,
                                                             activation_name_list, x)
        
# calculates the entropy of the activations
def calc_entropy(par_obj, prob_obj):
    """
    calculates discrete entropy and stores in probability object
    par_obj:
    prob_obj:
    """
    if ("weight" in par_obj.name):
        for key in par_obj.dic:
            p_T = prob_obj.p_layer_w[key][0]

            #initialise entry if not existing yet
            if(key not in prob_obj.w_entropy_dic.keys()):
                prob_obj.w_entropy_dic[key] = 0

            for p in p_T:
                # ignore 0 prop to allow logarithm
                if(p == 0):
                    continue
                prob_obj.w_entropy_dic[key] += -p*math.log(p,2)
    if ("output" in par_obj.name):
        for key in par_obj.dic:
            p_T = prob_obj.p_layer_act[key][0]

            #initialise entry if not existing yet
            if(key not in prob_obj.act_entropy_dic.keys()):
                prob_obj.act_entropy_dic[key] = 0
            for p in p_T:
                # ignore 0 prop to allow logarithm
                if(p == 0):
                    continue
                prob_obj.act_entropy_dic[key] += -p*math.log(p,2)



def calc_entropy_for_specific_t(cur_ts, px_i):
    """
    calculates conditional entropy for specific layer
    cur_ts: array of a layer for one epoch
    px_i: probability of the x ion which t is conditioned on
    returns: conditional entropy
    taken and adapted from:
   https://github.com/ravidziv/IDNNs/blob/master/idnns/information/mutual_information_calculation.py
    """
    unique_inverse_t, unique_counts = extract_unique(cur_ts)
    
    p_cur_ts = unique_counts / float(sum(unique_counts))
    p_cur_ts = np.asarray(p_cur_ts, dtype=np.float64).T
    H2X = px_i * (-np.sum(p_cur_ts * np.log2(p_cur_ts)))
    return H2X


def calc_condition_entropy(px, t_data, unique_inverse_x, par_flag):
    """
    starts calculation of conditional entropy for layer given input in parallel or sequential
    px: array of the input probabilities
    t_data: array matrix of layer for an epoch
    unique_inverse_x: inverse of the input vector
    par_flag: flag that decides whether parallel or sequential
    returns: conditional entropy
    taken and adapted from:
   https://github.com/ravidziv/IDNNs/blob/master/idnns/information/mutual_information_calculation.py
    """
    # Conditional entropy of t given x
    if par_flag:
        H2X_array = []
        H2X_array = np.array(
            Parallel(n_jobs=NUM_CORES)(delayed(calc_entropy_for_specific_t)
                                       (t_data[unique_inverse_x == i, :], px[i])
                                       for i in range(px.shape[0])))
        H2X = np.sum(H2X_array)
    else:
        H2X = 0
        for i in range(px.shape[0]):
            H2X += calc_entropy_for_specific_t(t_data[unique_inverse_x == i, :], px[i])
        
    return H2X


def calc_information(px, py, pt, data, unique_inverse_x, unique_inverse_y, par_flag):
    """
    calculates mutual information
    px: probability of inputs
    py: probability of outputs
    pt: layer probabilities
    data: layer data (activations)
    unique_inverse_x: inverse of continous input array
    unique_inverse_y: inverse of continous output array
    par_flag: flag that decides whether parallel or sequential
    returns: mutual information of layer with input and output
    """
    # Calculate the MI based on binned data
    H2 = -np.sum(pt * np.log2(pt))
    H2X = calc_condition_entropy(px, data, unique_inverse_x, par_flag)
    H2Y = calc_condition_entropy(py.T, data, unique_inverse_y, par_flag)
    IY = H2 - H2Y
    IX = H2 - H2X
    return IX, IY

def calc_information_between_in_out(px, py, x, y, unique_inverse_x, unique_inverse_y, par_flag):
    """
    calculates mutual information between input and output
    px: probability of inputs
    py: probability of outputs
    x: input array matrix
    y: output array matrix
    unique_inverse_x: inverse of the unique input arrays
    unique_inverse_y: inverse of the unique output arrays
    par_flag: flag that decides whether parallel or sequential
    returns: mutual information between input and output, entropy of input
    """
    HX = -np.sum(px * np.log2(px))
    HY = -np.sum(py * np.log2(py))
    H2X = calc_condition_entropy(px, y, unique_inverse_x, par_flag)
    H2Y = calc_condition_entropy(py, x, unique_inverse_y, par_flag)
    IX_Y = HX - H2Y
    IY_X = HY - H2X
    
    return IX_Y, HX



def calc_mutual_info(x, y, par_obj, prob_obj, bin_amount, bin_size_or_nr, par_flag):
    """
    governs the mutual information calculation
    reshapes images to continous array for calculations
    finds max and min for each layer
    calculates probabilities and entropy
    lastly calculated mutual information of layers with input and output
    x: input array matrix
    y: output array matrix
    par_obj: parameter object (output object)
    prob_obj: probability object to store prabilities, entropies, information... 
    bin_amount: amount OR size of bins
    bin_size_or_nr: flag that decides whether amount or size of bins gets applied
    par_flag: flag that decides whether parallel or sequential
    returns: probability object to store prabilities, entropies, information... 
    """
    # if image input reshape
    if len(x.shape) > 2:
        print("Reshaping tensor to arrays")
        x = np.reshape(x, (x.shape[0], x.shape[1] * x.shape[2] * x.shape[3]))

    if("output" in par_obj.name):
        par_obj.max, par_obj.min = find_max_min_activation(par_obj.dic)
        print("o max: ", par_obj.max)
        print("o min: ", par_obj.min)
    if("weight" in par_obj.name):
        find_max_min_weight(par_obj)
        print("w max: ", par_obj.max)
        print("w min: ", par_obj.min)
    
    #  calculate entropy and probabilities
    calc_probs(x, y, prob_obj, par_obj, bin_size_or_nr, bin_amount)
    calc_entropy(par_obj, prob_obj)
    # MI between input and output
    par_obj.mi_x_y, par_obj.entropy_x = calc_information_between_in_out(prob_obj.px, prob_obj.py, x,
                                                                        y, prob_obj.inverse_x,
                                                                        prob_obj.inverse_y,
                                                                        par_flag)
    
    print("X and Y MI: ", par_obj.mi_x_y, ", X Entropy: ", par_obj.entropy_x)
    
    # for each layer epoch combination calculate mutual infomration of layer iwth input and output
    for key in par_obj.dic.keys():
        if (key[1] == 0 and key[0]<=30) or (key[1] == 0 and key[0]%100 == 0):
            print("MI for epoch ", key[0],
                  " is being calculated for ", bin_amount, " bins")
        if("output" in par_obj.name):
            prob_obj.mi_x[key], prob_obj.mi_y[key] = calc_information(prob_obj.px, prob_obj.py,
                                                                      prob_obj.p_layer_act[key][0],
                                                                      par_obj.digitized[key][0],
                                                                      prob_obj.inverse_x,
                                                                      prob_obj.inverse_y,
                                                                      par_flag)
        if("weight" in par_obj.name):
            prob_obj.mi_x[key], prob_obj.mi_y[key] = calc_information(prob_obj.px, prob_obj.py,
                                                                      prob_obj.p_layer_w[key][0],
                                                                      par_obj.digitized[key][0],
                                                                      prob_obj.inverse_x,
                                                                      prob_obj.inverse_y,
                                                                      par_flag)
    
    return prob_obj
    