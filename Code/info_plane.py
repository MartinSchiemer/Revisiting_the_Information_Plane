"""
Author: Martin Schiemer
governs the information plane creation for various estimators
"""

import numpy as np
import multiprocessing
from joblib import Parallel, delayed

# EDGE
from EDGE.EDGE_4_3_1 import EDGE
from pprint import pprint

# KDE
import KDE

# npeet
import npeet

# other files
import binning
import plotting
import classes
import util
import early_stop_plotting

# counts CPUs for parallelization
CPUS = multiprocessing.cpu_count()


def MI_par_helper_dic(key, value, x, function, name, indicator_string):
    """
    helper function to parallelize edge and KSG estimator creating dictionaries that can
    then be united
    key: key of the output dictionary that holds activations
    value: activation for one layer and one epoch
    x: input or output data to calculate mutual information with
    function: estmator function
    name: name of the estimator
    indicator_string: flag if mutual information with input or output
    returns: dictionary with mutual information which will then be combined
    parallelization concept taken from:
    https://stackoverflow.com/questions/43200878/python-add-keyvalue-to-dictionary-in-parallelised-loop
    """
    dic = {}
    if "EDGE" in name:
        if indicator_string == "x":
            dic[key] = function(x, value, U=20, gamma=[1,1])
        elif indicator_string == "y":
            dic[key] = function(value, x, U=20, gamma=[1,0.1])
    if "Kraskov" in name:
        # tolist() as instructed in npeet github
        # not needed for activation for mixture mi
        # x already changed before
        if "dis" in name:
            dic[key] = function(x, value.tolist())
        elif "mix" in name:
                dic[key] = function(value, x, warning=False)
        elif "cont" in name:
            dic[key] = function(x, value.tolist())
    
    # print at certain intervals
    if (key[1] == 0 and key[0] < 30) or (key[1] == 0 and key[0]%50 == 0):
        print("Epoch " + str(key[0]) + " done")
    return dic

def MI_par_helper(x, y, par_object, function, name):
    """
    helper function to parallelize edge and KSG estimator
    x: input data to calculate mutual information with
    y: output data to calculate mutual information with
    function: estmator function
    name: name of the estimator
    returns: dictionary with all mutual information with intout and output
    parallelization concept taken from:
    https://stackoverflow.com/questions/43200878/python-add-keyvalue-to-dictionary-in-parallelised-loop
    """
    mi_x_dic = {}
    mi_y_dic = {}
    print("Starting " + name + " calculation for MI in parallel")
    with Parallel(n_jobs=CPUS) as parallel:
        # parallel code creates dictionary for each itteration and unites the dictionaries in one
        d_x = parallel(delayed(MI_par_helper_dic)(key, par_object.dic[key][0], x, function, name, "x")for key in par_object.dic.keys())
        print("calculated " + name + " MI_X in parallel")
        d_y = parallel(delayed(MI_par_helper_dic)(key, par_object.dic[key][0], y, function, name, "y")for key in par_object.dic.keys())
        print("calculated " + name + " MI_Y in parallel")
    # parallelization concept taken from:
    # https://stackoverflow.com/questions/43200878/python-add-keyvalue-to-dictionary-in-parallelised-loop
    mi_x_dic = {k: v for x in d_x for k, v in x.items()}
    mi_y_dic = {k: v for x in d_y for k, v in x.items()}
    
    return mi_x_dic, mi_y_dic


def Kolchinsky_par_helper(T, noise_variance, labelprobs, i, function):
    """
    helper function for KDE estimator parallelization
    """
    return labelprobs[i] * function(T, noise_variance)[0]


def Par_Kolchinsky_estimation(par_object, noise_variance, labelprobs, function,
                             label_indices):
    """
    parallel kde estimation and adds mutual information to dictionaries
    par_object: parameter object (output object)
    noise_variance: added noises variance
    labelprobs: probabilities of the class labels
    function: upper or lower KDE
    label_indices: array with indices of the different classes in the dataset
    returns: dictionaries with mutual information
    """
    print("Starting Kolchinsky calculation for MI in parallel")
    nats2bits = 1.0/np.log(2)
    dic_x = {}
    dic_y = {}
    with Parallel(n_jobs=CPUS) as parallel:
        for key in par_object.dic.keys():
            T = par_object.dic[key][0]
            entropy_T = function(T,noise_variance)[0]
            # Compute conditional entropies of layer activity given output
            entropy_T_giv_Y_array = []
            #parallelized calculation
            entropy_T_giv_Y_array = np.array(
                parallel(delayed(Kolchinsky_par_helper)(T[label_indices[i],:],
                                                                noise_variance, labelprobs,
                                                                     i, function)
                                   for i in label_indices.keys()))
            entropy_T_giv_Y = np.sum(entropy_T_giv_Y_array)
                        
            # Layer activity given input. This is simply the entropy of the Gaussian noise
            entropy_T_giv_X = KDE.kde_condentropy(T, noise_variance)
            
            dic_x[key] = nats2bits * (entropy_T - entropy_T_giv_X)
            dic_y[key] = nats2bits * (entropy_T - entropy_T_giv_Y)
        
        return dic_x, dic_y
            
def Kolchinsky_estimation(par_object, MI_object, labelprobs,
                         label_indices, higher_lower_flag, par_flag):
    """
    estimates mutual information using KDE either parallel or sequential
    par_object: parameter object (output object)
    MI_object: mutual information object
    labelprobs: probabilities of the class labels
    label_indices: array with indices of the different classes in the dataset
    higher_lower_flag: flag that decides whether higher or lower bound KDE is used
    par_flag: flag that decides whether prallel or sequential
    returns: mutual information object
    partly taken from:
    https://github.com/artemyk/ibsgd
    """
    noise_variance = 1e-3
    # nats to bits conversion factor
    nats2bits = 1.0/np.log(2)
    if higher_lower_flag == True:
        KDE_estimator_func = KDE.entropy_estimator_kl
    else:
        KDE_estimator_func = KDE.entropy_estimator_bd
    
    if par_flag:
        MI_object.mi_x, MI_object.mi_y = Par_Kolchinsky_estimation(par_object, noise_variance,
                                                                   labelprobs, KDE_estimator_func,
                                                                   label_indices)
    else:
        for key in par_object.dic.keys():
            T = par_object.dic[key][0]
            entropy_T = KDE_estimator_func(T,noise_variance)[0]
            # Compute conditional entropies of layer activity given output
            entropy_T_giv_Y = 0.
            for i in label_indices.keys():
                entropy_cond = KDE_estimator_func(T[label_indices[i],:],noise_variance)[0]
                entropy_T_giv_Y += labelprobs[i] * entropy_cond
            # Layer activity given input. This is simply the entropy of the Gaussian noise
            entropy_T_giv_X = KDE.kde_condentropy(T, noise_variance)  
            
            MI_object.mi_x[key] = nats2bits * (entropy_T - entropy_T_giv_X)
            MI_object.mi_y[key] = nats2bits * (entropy_T - entropy_T_giv_Y)
            if key[1] == 1 and (key[0]%50 == 0 or key[0] <= 30):
                print("calculated KDE MI_X and MI_Y for epoch:", key[0])
    return MI_object


def EDGE_estimation(x, y, par_object, MI_object, par_flag):
    """
    estimates mutual information using EDGE either parallel or sequential
    x: input data
    y: output data
    par_object: parameter object (output object)
    MI_object: mutual information object
    par_flag: flag that decides whether prallel or sequential
    returns: mutual information object
    """
    # compute EDGE MI
    if par_flag == True:
        MI_object.mi_x, MI_object.mi_y = MI_par_helper(x, y, par_object, EDGE, "EDGE")
    else:
        # procedural calculations
        # compute MI for all recorded epochs
        for key in par_object.dic.keys():
            T = par_object.dic[key][0]
            # gamma set to one since the data is continous
            MI_object.mi_x[key] = EDGE(x, T, U=20, gamma=[1,1])
            # y gamma set to 0.1 since categorical is not continous
            MI_object.mi_y[key] = EDGE(T, y, U=20, gamma=[1,0.1])
            if key[1] == 1 and (key[0]%50 == 0 or key[0] <= 30):
                print("calculated EDGE MI_X and MI_Y procedural for epoch:", key[0])

    return MI_object


def Kras_estimation(x, y, par_object, MI_object, discrete_flag, par_flag):
    """
    estimates mutual information using KSE either parallel or sequential
    x: input data
    y: output data
    par_object: parameter object (output object)
    MI_object: mutual information object
    discrete_flag: flag that decides whether discrete, mix or continous KSG is used
    par_flag: flag that decides whether prallel or sequential
    returns: mutual information object
    """
    # compute MI for all recorded epochs
    if par_flag == True:
        if discrete_flag == 1:
            MI_object.mi_x, MI_object.mi_y = MI_par_helper(x.tolist(), y.tolist(), par_object, 
                                                           npeet.midd, "Kraskov_dis")
        if discrete_flag == 2:
            MI_object.mi_x, MI_object.mi_y = MI_par_helper(x.tolist(), y.tolist(), par_object, 
                                                           npeet.micd, "Kraskov_mix")
        if discrete_flag == 3:
            MI_object.mi_x, MI_object.mi_y = MI_par_helper(x.tolist(), y.tolist(), par_object, 
                                                           npeet.mi, "Kraskov_cont")
    else:
        # procedural code
        for key in par_object.dic.keys():
            T = par_object.dic[key][0]
            # gamma set to one since the data is continous
            if discrete_flag == 1:
                MI_object.mi_x[key] = npeet.midd(x.tolist(), T.tolist())
                MI_object.mi_y[key] = npeet.midd(T.tolist(), y.tolist())
            if discrete_flag == 2:
                MI_object.mi_x[key] = npeet.micd(T, x.tolist())
                MI_object.mi_y[key] = npeet.micd(T, y.tolist())
            if discrete_flag == 3:
                MI_object.mi_x[key] = npeet.mi(T.tolist(), x.tolist())
                MI_object.mi_y[key] = npeet.mi(y.tolist(), T.tolist())

            if key[0]%100 == 0 or key[0] <= 30:
                print("calculated Kraskov MI for (epoch, layer):", key)

    return MI_object


def create_infoplane(name, x, y, par_object, est_type_flag,
                     color_l, bin_amount=0.07, bin_size_or_nr=True,
                     show_flag=False, separate_flag=False, save_flag=True, par_flag=True):
    """
    starts mutual information calculation and then plots results
    name: name of the network
    x: input data
    y: output data
    par_object: parameter object (output object)
    est_type_flag: flag that decides whcih estimator is used
    color_l: list of colours for the different layers for the plot
    bin_amount: bin size OR number of bins
    bin_size_or_nr: flag that decides if bin size OR number is used
    show_flag: falg that decides if plot are displayed 
    separate_flag: 
    save_fla: falg that decides if plots and data is saved
    par_flag: flag that decides whether prallel or sequential
    returns: mutual information object
    """
    # if image input transform to continous features
    if len(x.shape) > 2:
        print("Reshaping tensor to arrays")
        x = np.reshape(x, (x.shape[0], x.shape[1] * x.shape[2] * x.shape[3]))
        
    # if weights repeat weights vector to equal size of training samples
    if par_object.name == "weights":
        for key in par_object.dic.keys():
            if key[0]%100 == 0:
                print(key)
            ws = par_object.dic[key][0]
            par_object.dic[key][0] = np.repeat(ws.reshape(1, ws.shape[0]*ws.shape[1]),
                                                x.shape[0],axis=0)
        
    
    # depending on est_type_flag value use different estimation option
    # 1: Binning, 2: EDGE, 3: KDE upper, 4: KDE lower, 5: KSG discrete,
    # 5: KSG mix, 5: KSG continous,
    if est_type_flag in [1,2,3,4,5,6,7]:
        if est_type_flag == 1:
            name = name + "_Prob" + str(bin_amount) +"_Bins"
            MI_object = classes.Binning_MI()
            MI_object = binning.calc_mutual_info(x, y, par_object,
                                                          MI_object, bin_amount, bin_size_or_nr,
                                                          par_flag)
        elif est_type_flag == 2:
            name = name + "_MI_EDGE"
            MI_object = classes.EDGE_MI()
            MI_object = EDGE_estimation(x, y, par_object, MI_object, par_flag)
        elif est_type_flag in [3,4]:
            name = name + "_MI_KDE"
            MI_object = classes.KDE_MI()
            labelprobs = np.mean(y, axis=0)
            # Save indexes of train data for each of the output classes
            label_indices = {}
            for i in range(y.shape[1]):
                 # return make non categorical to extract indices
                 # from: https://github.com/keras-team/keras/issues/4981
                 label_indices[i] = np.argmax(y, axis=1) == i
            
            if est_type_flag == 3:
                name = name + "upper"
                MI_object = Kolchinsky_estimation(par_object, MI_object, labelprobs,
                                         label_indices, True, par_flag)
            if est_type_flag == 4:
                name = name + "lower"
                MI_object = Kolchinsky_estimation(par_object, MI_object, labelprobs,
                                         label_indices, False, par_flag)
        elif est_type_flag == 5:
            name = name + "_MI_Kras_disc"
            MI_object = classes.Kras_MI()
            MI_object = Kras_estimation(x, y, par_object, MI_object, 1, par_flag)
        elif est_type_flag == 6:
            name = name + "_MI_Kras_mix"
            MI_object = classes.Kras_MI()
            MI_object = Kras_estimation(x, y, par_object, MI_object, 2, par_flag)
        elif est_type_flag == 7:
            name = name + "_MI_Kras_cont"
            MI_object = classes.Kras_MI()
            MI_object = Kras_estimation(x, y, par_object, MI_object, 3, par_flag)    
            
        
        # add parameters to object
        MI_object.act_func = par_object.act_func
        MI_object.batchsize = par_object.batchsize
        # final model testscore
        score = par_object.int_model_score[np.amax(list(par_object.int_model_score.keys()))]  
        MI_object.model_score = score
        
        # if neither show nor save just return object
        if save_flag == False and show_flag == False:
            return MI_object
        
        # save MI data if required
        if save_flag == True:
            util.save(MI_object, name)
            
        
        # create plots
        plotting.plot_info_plane(MI_object, name, separate_flag, color_l, show_flag, save_flag)
        
        return MI_object
        
    else:
        print("Sorry please choose a valid estimation type: 1=Binning, 2=EDGE, 3=Kocholsky upper, 4=Kocholsky lower, 5=Kraskov for discrete variables, 6=Kraskov for mixed variables, 7=Kraskov for continous variables")

        
def ranged_create_infoplane(max_x, max_y, name, x, y, par_object, est_type_flag,
                     color_l, bin_amount=0.07, bin_size_or_nr=True,
                     show_flag=False, separate_flag=False, save_flag=True, par_flag=True):
    """
    calculates information plane and then plots it in certain range
    useful for e.g. early stop plot comparison where the bounds are the ones of full epochs
    max_x: max x value
    max_y: max y value
    x: input data
    y: output data
    par_object: parameter object (output object)
    est_type_flag: flag that decides whcih estimator is used
    color_l: list of colours for the different layers for the plot
    bin_amount=0.07: 
    bin_size_or_nr:
    show_flag: falg that decides if plot are displayed 
    save_fla: falg that decides if plots and data is saved
    par_flag: flag that decides whether prallel or sequential
    returns: mutual information object
    """
    
    if est_type_flag == 1:
        name = name + "_Prob" + str(bin_amount) +"_Bins"
        MI_object = classes.Binning_MI()
    elif est_type_flag == 2:
        name = name + "_MI_EDGE"
        MI_object = classes.EDGE_MI()
    elif est_type_flag in [3,4]:
        name = name + "_MI_KDE"
        MI_object = classes.KDE_MI()
    elif est_type_flag == 5:
        name = name + "_MI_Kras_disc"
        MI_object = classes.Kras_MI()
    elif est_type_flag == 6:
        name = name + "_MI_Kras_mix"
        MI_object = classes.Kras_MI()
    elif est_type_flag == 7:
        name = name + "_MI_Kras_cont"
        MI_object = classes.Kras_MI()
    
    from numpy.random import seed
    seed(1337)
    from tensorflow import set_random_seed
    set_random_seed(1337)
    
    MI_object = create_infoplane(name, x, y, par_object, est_type_flag,
                                 color_l, bin_amount, bin_size_or_nr,
                                 True, separate_flag, False, par_flag)
    # save MI data if required
    if save_flag == True:
        util.save(MI_object, name)
    
    early_stop_plotting.max_fix_plot_info_plane_combination_view(MI_object, max_x, max_y,
                                                                     name, color_l, show_flag,
                                                                     save_flag)               
        
    return MI_object
    
    