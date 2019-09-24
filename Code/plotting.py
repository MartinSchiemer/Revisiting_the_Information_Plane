"""
Author: Martin Schiemer
provides plotting functionalitites
"""

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn as sns
import os
import sys
import math


def plot_bin_histo(name, layer_nr, outputs_obj, mi_obj, show_flag, save_flag, limit = False):
    """
    plots histogram of the bins after binning estimation
    name: name of the network
    layer_nr: index of the layer which should be plotted
    outputs_obj: output object which holds the activation dictionary
    mi_obj: mutual information object
    show_flag: flag that decides if plot should be displayed
    save_flag: flag that decides if plot should be saved
    limit: flag that decides if the plot range should be limited
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, sharey=False)
    fig.set_figheight(10)
    fig.set_figwidth(15)
    
    #fig.suptitle(("Empty bins development for layer " + str(layer_nr) + " (" + name + ", score: "+ str(mi_obj.model_score) + ")"))
    fig.suptitle(("Empty bins development for layer " + str(layer_nr) + " (score: "+ str(mi_obj.model_score) + ")"), fontsize=15)
    
    ax3.set_xlabel("Bins")
    ax1.set_ylabel("Amount in bin")
    ax2.set_ylabel("Amount in bin")
    ax3.set_ylabel("Amount in bin")
    ax1.set_title("Bins at epoch 0")
    ax2.set_title("Bins at half of the epochs")
    ax3.set_title("Bins at the last epoch")
    
    layer_keys = [key for key in outputs_obj.digitized.keys() if key[1] == layer_nr]
    
    interval_keys = [layer_keys[0], layer_keys[len(layer_keys)//2], layer_keys[-1]]
    
    
    
    ax1.hist([x for y in outputs_obj.digitized[interval_keys[0]][0] for x in y],
             mi_obj.bin_edges[interval_keys[0]])
    ax2.hist([x for y in outputs_obj.digitized[interval_keys[1]][0] for x in y],
             mi_obj.bin_edges[interval_keys[1]])
    ax3.hist([x for y in outputs_obj.digitized[interval_keys[2]][0] for x in y],
             mi_obj.bin_edges[interval_keys[2]])
    
    if limit == True:
        ax1.set_xlim(left=0.1, right=None)
        ax2.set_xlim(left=0.1, right=None)
        ax3.set_xlim(left=0.1, right=None)
        ax1.set_ylim(top= 150)
        ax2.set_ylim(top= 150)
        ax3.set_ylim(top= 150)

    if save_flag == True:
        if not os.path.exists("Results/Plots/EmptyBins/"):
                try:
                    os.makedirs("Results/Plots/EmptyBins/")
                except OSError as error:
                    if error.errno != errno.EEXIST:
                        raise
        plt.savefig("Results/Plots/EmptyBins/" + name + "_layer" + str(layer_nr) + "_BinHisto.png")
        
    if show_flag == True:
        plt.show()
    else:
        plt.close()


def plot_empty_bins(name, MI_obj, max_epochs, color_list, show_flag, save_flag, full_flag=True):
    """
    plots plot of empty bins in % after binning estimation
    name: name of the network
    MI_obj: mutual information object
    max_epochs: maximum nr of epochs
    color_list: list of colours for different layers
    show_flag: flag that decides if plot should be displayed
    save_flag: flag that decides if plot should be saved
    full_flag: flag that decides if plot has 1 or 2 axis (1 for full epoch 2 for full epoch
               and part)
    """
    print("Plotting empty bins")
    if full_flag == False:
        fig, (ax1, ax2) = plt.subplots(2, 1, sharey=True)
    else:
        fig, ax1 = plt.subplots()
    fig.set_figheight(10)
    fig.set_figwidth(15)
    
    #fig.suptitle(("Empty bins development (" + name + ", score: "+ str(MI_obj.model_score) + ")"))
    fig.suptitle(("Empty bins development (score: "+ str(MI_obj.model_score) + ")"), fontsize=15)
    
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Amount of empty bins in %")

    layer_nrs = [key[1] for key in MI_obj.unused_bins.keys()]
    max_layer = np.amax(layer_nrs)
    activation_name_list = MI_obj.act_func

    label_count = 0
    for key in MI_obj.unused_bins.keys():
        if full_flag == False:
            if label_count < len(activation_name_list):
                ax1.scatter(key[0], (MI_obj.unused_bins[key]/MI_obj.tot_bins[key])*100,
                            color=color_list[key[1]], label="l"+str(key[1]+1)+ " " +activation_name_list[key[1]])
                ax2.scatter(key[0], (MI_obj.unused_bins[key]/MI_obj.tot_bins[key])*100,
                            color=color_list[key[1]], label="l"+str(key[1]+1)+ " " +activation_name_list[key[1]])
            else:
                ax1.scatter(key[0], (MI_obj.unused_bins[key]/MI_obj.tot_bins[key])*100,
                            color=color_list[key[1]])
                ax2.scatter(key[0], (MI_obj.unused_bins[key]/MI_obj.tot_bins[key])*100,
                            color=color_list[key[1]])
        else:
            if label_count < len(activation_name_list):
                ax1.scatter(key[0], (MI_obj.unused_bins[key]/MI_obj.tot_bins[key])*100,
                            color=color_list[key[1]], label="l"+str(key[1]+1)+ " " +activation_name_list[key[1]])
            else:
                ax1.scatter(key[0], (MI_obj.unused_bins[key]/MI_obj.tot_bins[key])*100,
                            color=color_list[key[1]])
        label_count += 1
        
    
        
    ax1.legend()
    if full_flag == False:
        ax2.set_xlim(left=0, right=max_epochs)
    ax1.set_xlim(left=0, right=None)
    ax1.set_ylim(bottom=0, top=None)
    
    remove_neg_ticks(ax1, "x")
    remove_neg_ticks(ax1, "y")
    
    plt.tight_layout()
    if save_flag == True:
        if not os.path.exists("Results/Plots/EmptyBins/"):
                try:
                    os.makedirs("Results/Plots/EmptyBins/")
                except OSError as error:
                    if error.errno != errno.EEXIST:
                        raise
        plt.savefig("Results/Plots/EmptyBins/" + name + "_EmptyBins.png")
        
    if show_flag == True:
        plt.show()
    else:
        plt.close()


def plot_std_abs_activations(name, activation_obj, color_list, max_epochs,
                                  show_flag, save_flag, full_flag):
    """
    plots absolute standard deviation of the activations
    name: name of the network
    activation_obj: output object that holds the activations
    color_list: list of colours for different layers
    max_epochs: maximum nr of epochs
    show_flag: flag that decides if plot should be displayed
    save_flag: flag that decides if plot should be saved
    full_flag: flag that decides if plot has 1 or 2 axis (1 for full epoch 2 for full epoch
               and part)
    """
    print("Creating standard deviation development plot")
    if full_flag == False:
        fig, (ax1, ax2) = plt.subplots(2, 1, sharey=True)
    else:
        fig, ax1 = plt.subplots()
    fig.set_figheight(10)
    fig.set_figwidth(15)
    plt.subplots_adjust(
                        top = 0.94,
                        wspace = 0.1, 
                        )
    
    #fig.suptitle(("Mean and Standard Deviation (" + name + ", score: "+ str(activation_obj.model_score) + ")"))
    fig.suptitle(("Mean and Standard Deviation, score: "+ str(activation_obj.model_score) + ")"), fontsize=15)
    
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Standard Deviation")


    layer_nrs = [key[1] for key in activation_obj.dic.keys()]
    max_layer = np.amax(layer_nrs)
    activation_name_list = activation_obj.act_func
    output_layers_with05_clustered_std = ["sigmoid", "tanh", "softmax"]
    
    label_count = 0
    for key in activation_obj.dic.keys():
        if full_flag == False:
            if (key[1] == max_layer and
                activation_name_list[key[1]] in output_layers_with05_clustered_std):
                if label_count < len(activation_name_list):
                    # -.5 offset to concentrate the 0 and 1 cluster at one point after
                    # taking the absolute value
                    ax1.scatter(key[0], np.std(abs(activation_obj.dic[key][0]-0.5)),
                            color=color_list[key[1]], label="l"+str(key[1]+1)+ " " +activation_name_list[key[1]])
                    ax2.scatter(key[0], np.std(abs(activation_obj.dic[key][0]-0.5)),
                            color=color_list[key[1]], label="l"+str(key[1]+1)+ " " +activation_name_list[key[1]])
                else:
                    ax1.scatter(key[0], np.std(abs(activation_obj.dic[key][0]-0.5)),
                            color=color_list[key[1]])
                    ax2.scatter(key[0], np.std(abs(activation_obj.dic[key][0]-0.5)),
                            color=color_list[key[1]])
            else:
                if label_count < len(activation_name_list):
                    ax1.scatter(key[0], np.std(abs(activation_obj.dic[key][0])),
                            color=color_list[key[1]], label="l"+str(key[1]+1)+ " " +activation_name_list[key[1]])
                    ax2.scatter(key[0], np.std(abs(activation_obj.dic[key][0])),
                            color=color_list[key[1]], label="l"+str(key[1]+1)+ " " +activation_name_list[key[1]])
                else:
                    ax1.scatter(key[0], np.std(abs(activation_obj.dic[key][0])),
                            color=color_list[key[1]])
                    ax2.scatter(key[0], np.std(abs(activation_obj.dic[key][0])),
                            color=color_list[key[1]])
        else:
            if (key[1] == max_layer and
                activation_name_list[key[1]] in output_layers_with05_clustered_std):
                if label_count < len(activation_name_list):
                    # -.5 offset to concentrate the 0 and 1 cluster at one point after
                    # taking the absolute value
                    ax1.scatter(key[0], np.std(abs(activation_obj.dic[key][0]-0.5)),
                            color=color_list[key[1]], label="l"+str(key[1]+1)+ " " +activation_name_list[key[1]])
                else:
                    ax1.scatter(key[0], np.std(abs(activation_obj.dic[key][0]-0.5)),
                            color=color_list[key[1]])
            else:
                if label_count < len(activation_name_list):
                    ax1.scatter(key[0], np.std(abs(activation_obj.dic[key][0])),
                            color=color_list[key[1]], label="l"+str(key[1]+1)+ " " +activation_name_list[key[1]])
                else:
                    ax1.scatter(key[0], np.std(abs(activation_obj.dic[key][0])),
                            color=color_list[key[1]])
        label_count +=1
            
            
    ax1.legend()
    if full_flag == False:
        ax2.set_xlim(left=0, right=max_epochs)
    ax1.set_xlim(left=0, right=None)
    ax1.set_ylim(bottom=0, top=None)
    remove_neg_ticks(ax1, "x")
    remove_neg_ticks(ax1, "y")

    plt.tight_layout()
    if save_flag == True:
        if not os.path.exists("Results/Plots/MeanSTD/"):
                try:
                    os.makedirs("Results/Plots/MeanSTD/")
                except OSError as error:
                    if error.errno != errno.EEXIST:
                        raise
        plt.savefig("Results/Plots/MeanSTD/" + name + "_MeanSTD.png")
        
    if show_flag == True:
        plt.show()
    else:
        plt.close()
    
    

def remove_neg_ticks(ax, x_or_y):
    """
    removes negative ticks of plot axis
    ax: axis where neg ticks hould be removed
    x_or_y: flag if x or y axis
    """
    if x_or_y == "x":
        axticks = [tick for tick in ax.get_xticks() if tick >=0]
        ax.set_xticks(axticks)
    if x_or_y == "y":
        axticks = [tick for tick in ax.get_yticks() if tick >=0]
        ax.set_yticks(axticks)


def plot_test_development(test_score_dic, name, show_flag, save_flag):
    """
    plots test score development
    test_score_dic: dictionary with test scores
    name: name of the network
    show_flag: flag that decides if plot should be displayed
    save_flag: flag that decides if plot should be saved
    """
    if not test_score_dic:
        print("testscore has not been recorded")
    else:
        print("creating testscore devolopment plot")

        cmap = plt.get_cmap('gnuplot')
        last_it = np.amax(list(test_score_dic.keys()))
        colors = [cmap(i) for i in np.linspace(0, 1, last_it + 1)]

        fig, ax1 = plt.subplots()
        fig.set_figheight(10)
        fig.set_figwidth(15)
        #ax1.set_title(("Test score per epoch (" + name + ", last epoch: " + str(last_it) + ")"))
        ax1.set_title(("Test score per epoch (last epoch: " + str(last_it) + ")"), fontsize=15)
        
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Test Score")

        for key in test_score_dic:
            ax1.scatter(key, test_score_dic[key], color=colors[key])
            
        #ax1.set_xlim(left=0, right=None)
        ax1.set_ylim(bottom=0, top=None)
        #ax1.set_xbound(lower=-0.05)
        ax1.set_ybound(lower=-0.05)
        #remove_neg_ticks(ax1, "x")
        remove_neg_ticks(ax1, "y")
        #ax1.set_xticks(ax1xticks)
        #ax1.set_xticks(ax1xticks)
        
        #fig.tight_layout()
        plt.tight_layout()
        if save_flag == True:
            if not os.path.exists("Results/Plots/Testscore/"):
                try:
                    os.makedirs("Results/Plots/Testscore/")
                except OSError as error:
                    if error.errno != errno.EEXIST:
                        raise
            plt.savefig("Results/Plots/Testscore/" + name + "_testscore.png")
        if show_flag == True:
            plt.show()
        else:
            plt.close()


def plot_history(h_object, name, show_flag, save_flag):
    """
    plots traning and validation loss development
    h_object: keras history object
    name: name of the network
    show_flag: flag that decides if plot should be displayed
    save_flag: flag that decides if plot should be saved
    """
    print("creating history plot")
    keys = [*h_object.keys()]
    last_it = len(h_object["loss"]) - 1
    # compares the loss and metric for equality
    list_compare = all( math.isclose(e1, e2, abs_tol=0.1) for e1, e2 in
                             zip(h_object[keys[0]], h_object[keys[1]]))
    # if loss and metric are the same only plot 1
    if list_compare:
        fig, ax1 = plt.subplots(1, 1)
        fig.set_figheight(10)
        fig.set_figwidth(15)

        # summarize history for loss
        ax1.plot(h_object['loss'])
        ax1.plot(h_object['val_loss'])
        #ax1.set_title('Model Loss (' + name + " score: " +
        #              str(h_object["model_score"]) + ", last epoch: " + str(last_it + 1) + ")" )
        ax1.set_title("Model Loss (score: " +
                      str(h_object["model_score"]) + ", last epoch: " + str(last_it + 1) + ")", fontsize=15)
        
        ax1.set_ylabel('loss')
        ax1.set_xlabel('epoch')
        ax1.legend(['train', 'validation'], loc='upper left')
        
        ax1.set_xlim(left=0, right=None)
        ax1.set_ylim(bottom=0, top=None)
        ax1.set_xbound(lower=-0.05)
        ax1.set_ybound(lower=-0.05)
        remove_neg_ticks(ax1, "x")
        remove_neg_ticks(ax1, "y")
        
        #ax1.set_xticks(ax1xticks)
    
    else:    
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        fig.set_figheight(10)
        fig.set_figwidth(15)

        # summarize history for accuracy
        ax1.plot(h_object[keys[1]])
        ax1.plot(h_object[keys[3]])
        ax1.set_title(('model ' + str(keys[1]) + "(score: " + str(h_object["model_score"]) + ")"))
        ax1.set_ylabel(keys[1])
        ax1.set_xlabel('epoch')
        ax1.legend(['train', 'validation'], loc='upper left')

        # summarize history for loss
        ax2.plot(h_object['loss'])
        ax2.plot(h_object['val_loss'])
        ax2.set_title('model loss')
        ax2.set_ylabel('loss')
        ax2.set_xlabel('epoch')
        ax2.legend(['train', 'validation'], loc='upper left')
        
#         #ax1.set_xlim(left=0-0.5, right=None)
        ax1.set_ylim(bottom=0, top=None)
#         #ax2.set_xlim(left=0-0.5, right = None)
        ax2.set_ylim(bottom=0, top=None)
        ax1.set_ybound(lower=-0.05)
        ax2.set_ybound(lower=-0.05)
        remove_neg_ticks(ax1, "y")
        remove_neg_ticks(ax2, "y")
    
    #fig.tight_layout()
    plt.tight_layout()
    if save_flag == True:
        if not os.path.exists("Results/Plots/History/"):
                try:
                    os.makedirs("Results/Plots/History/")
                except OSError as error:
                    if error.errno != errno.EEXIST:
                        raise
        plt.savefig("Results/Plots/History/" + name + "_loss.png")
    if show_flag == True:
        plt.show()
    else:
        plt.close()
    


def plot_separate_info_plane_layer_view(MI_object, name, color_l, show_flag, save_flag):
    """
    plots information plane separate into different layers
    MI_object: mutual information object
    name: name of the network
    color_l: list of colours for different layers
    show_flag:flag that decides if plot should be displayed
    save_flag: flag that decides if plot should be saved
    """
    print("creating separate info plane layer view plot")
    activations = MI_object.act_func
    fig, axes = plt.subplots(len(activations),2,sharex=True,sharey=True)

    fig.set_figheight(30)
    fig.set_figwidth(15)
    plt.subplots_adjust(
                        top = 0.97,
                        wspace = 0.05, 
                        )

    #fig.suptitle(("Information Plane (" + name + ", score: " + str(MI_object.model_score) + ")"))
    fig.suptitle(("Information Plane (score: " + str(MI_object.model_score) + ")"), fontsize=15)

    color_list = color_l
    cmap = plt.get_cmap('gnuplot')
    last_it = np.amax(list(MI_object.mi_x.keys()))
    colors = [cmap(i) for i in np.linspace(0, 1, last_it + 1)]

    # controls start and stop sign
    label_count = 0
    sp_label_count = 0

    for key in MI_object.mi_x.keys():
        # epochview
        axes[key[1],1].plot(MI_object.mi_x[key], MI_object.mi_y[key],marker="o",
                              markersize=9, linewidth=0.2, color=colors[key[0]])
            
        # layerview
        if key[0] == 0:
            if sp_label_count == 0:
                axes[key[1],0].scatter(MI_object.mi_x[key], MI_object.mi_y[key],
                       color='black', linewidth=5, label='start')
                sp_label_count += 1
            else:
                axes[key[1],0].scatter(MI_object.mi_x[key], MI_object.mi_y[key],
                       color='black', linewidth=5)
        elif key[0] == list(MI_object.mi_x.keys())[-1][0]:
            if sp_label_count == 1:
                axes[key[1],0].scatter(MI_object.mi_x[key], MI_object.mi_y[key],
                       color='black', linewidth=5, marker="v", label='end')
                sp_label_count += 1
            else:
                axes[key[1],0].scatter(MI_object.mi_x[key], MI_object.mi_y[key],
                       color='black', linewidth=5, marker="v")
        else:
            if label_count < len(activations):
                axes[key[1],0].scatter(MI_object.mi_x[key],
                       MI_object.mi_y[key], color=color_list[key[1]], label="l"+str(key[1]+1)+ " " +activations[key[1]])
                label_count += 1
            else:
                axes[key[1],0].scatter(MI_object.mi_x[key],
                       MI_object.mi_y[key], color=color_list[key[1]])

    # unify axes to start from 0            
    for i in range(len(activations)):
        for j in range(2):
            axes[i,j].set_xlabel("I(X;T)")
            axes[i,j].set_ylabel("I(T;Y)")
            axes[i,j].set_xlim(left=0, right=None)
            axes[i,j].set_ylim(bottom=0, top=None)
            #axes[i,j].set_xbound(lower=-0.05)
            #axes[i,j].set_ybound(lower=-0.05)
            remove_neg_ticks(axes[i,j], "x")
            remove_neg_ticks(axes[i,j], "y")
    
    #fig.tight_layout()
    plt.tight_layout()
    if save_flag == True:
        if not os.path.exists("Results/Plots/LayerviewSplit/"):
                try:
                    os.makedirs("Results/Plots/LayerviewSplit/")
                except OSError as error:
                    if error.errno != errno.EEXIST:
                        raise
        plt.savefig("Results/Plots/LayerviewSplit/" + name + "_layerviewsplit.png")
    if show_flag == True:
        plt.show()
    else:
        plt.close()
        

def plot_info_plane_layer_view(MI_object, name, color_l, show_flag, save_flag):
    """
    plots information plane in a layer view
    MI_object: mutual information object
    name: name of the network
    color_l: list of colours for different layers
    show_flag:flag that decides if plot should be displayed
    save_flag: flag that decides if plot should be saved
    """
    print("creating info plane layer view plot")
    fig, ax = plt.subplots()
    fig.set_figheight(10)
    fig.set_figwidth(15)

    #ax.set_title(("Information Plane (" + name + ", score: " + str(MI_object.model_score) + ")"))
    ax.set_title(("Information Plane (score: " + str(MI_object.model_score) + ")"), fontsize=15)
    

    ax.set_xlabel("I(X;T)")
    ax.set_ylabel("I(T;Y)")

    color_list = color_l
    activations = MI_object.act_func
    
    label_count = 0
    sp_label_count = 0
    for key in MI_object.mi_x.keys():
        if key[0] == 0:
            if sp_label_count == 0:
                ax.scatter(MI_object.mi_x[key], MI_object.mi_y[key],
                       color='black', linewidth=5, label='start')
                sp_label_count += 1
            else:
                ax.scatter(MI_object.mi_x[key], MI_object.mi_y[key],
                       color='black', linewidth=5)
        elif key[0] == list(MI_object.mi_x.keys())[-1][0]:
            if sp_label_count == 1:
                ax.scatter(MI_object.mi_x[key], MI_object.mi_y[key],
                       color='black', linewidth=5, marker="v", label='end')
                sp_label_count += 1
            else:
                ax.scatter(MI_object.mi_x[key], MI_object.mi_y[key],
                       color='black', linewidth=5, marker="v")
        else:
            if label_count < len(activations):
                ax.scatter(MI_object.mi_x[key],
                       MI_object.mi_y[key], color=color_list[key[1]], label="l"+str(key[1]+1)+ " " +activations[key[1]])
                label_count += 1
            else:
                ax.scatter(MI_object.mi_x[key],
                       MI_object.mi_y[key], color=color_list[key[1]])

        ax.legend()
    
    ax.set_xlim(left=0, right=None)
    ax.set_ylim(bottom=0, top=None)
    ax.set_xbound(lower=-0.05)
    ax.set_ybound(lower=-0.05)
    remove_neg_ticks(ax, "x")
    remove_neg_ticks(ax, "y")
    
    #fig.tight_layout()
    plt.tight_layout()
    if save_flag == True:
        if not os.path.exists("Results/Plots/Layerview/"):
                try:
                    os.makedirs("Results/Plots/Layerview/")
                except OSError as error:
                    if error.errno != errno.EEXIST:
                        raise
        plt.savefig("Results/Plots/Layerview/" + name + "_layerview.png")
    if show_flag == True:
        plt.show()
    else:
        plt.close()
    
        
def plot_info_plane_epoch_view(MI_object, name, show_flag, save_flag):
    """
    plots information plane in an epoch view
    MI_object: mutual information object
    name: name of the network
    color_l: list of colours for different layers
    show_flag:flag that decides if plot should be displayed
    save_flag: flag that decides if plot should be saved
    """
    print("creating info plane epoch view plot")
    fig, ax = plt.subplots()
    fig.set_figheight(10)
    fig.set_figwidth(15)

    #ax.set_title(("Information Plane (" + name + ", score: "+ str(MI_object.model_score) + ")"))
    ax.set_title(("Information Plane (score: "+ str(MI_object.model_score) + ")"), fontsize=15)
    ax.set_xlabel("I(X;T)")
    ax.set_ylabel("I(T;Y)")
    
    activations = MI_object.act_func
    cmap = plt.get_cmap('gnuplot')
    
    last_it = np.amax(list(MI_object.mi_x.keys()))
    colors = [cmap(i) for i in np.linspace(0, 1, last_it + 1)]
    
    
    mi_x_list = []
    mi_y_list = []
    act_count = 0
    for key in MI_object.mi_x.keys():
        if act_count < len(activations):
            mi_x_list.append(MI_object.mi_x[key])
            mi_y_list.append(MI_object.mi_y[key])
            act_count += 1
            if act_count == len(activations):
                c = colors[key[0]]
                ax.plot(mi_x_list, mi_y_list, marker="o",
                        markersize=9, linewidth=0.2, color=c)
                act_count = 0
                mi_x_list = []
                mi_y_list = []
                
    ax.set_xlim(left=0, right=None)
    ax.set_ylim(bottom=0, top=None)
    ax.set_xbound(lower=-0.05)
    ax.set_ybound(lower=-0.05)
    remove_neg_ticks(ax, "x")
    remove_neg_ticks(ax, "y")
    
    #fig.tight_layout()
    plt.tight_layout()
    if save_flag == True:
        if not os.path.exists("Results/Plots/Epochview/"):
                try:
                    os.makedirs("Results/Plots/Epochview/")
                except OSError as error:
                    if error.errno != errno.EEXIST:
                        raise
        plt.savefig("Results/Plots/Epochview/" + name + "_epochview.png")
        
    if show_flag == True:
        plt.show()
    else:
        plt.close()
        
def plot_info_plane_combination_view(MI_object, name, color_l, show_flag, save_flag):
    """
    plots information plane in a combination of epoch and layer view
    MI_object: mutual information object
    name: name of the network
    color_l: list of colours for different layers
    show_flag:flag that decides if plot should be displayed
    save_flag: flag that decides if plot should be saved
    """
    print("Creating combinationview plot")
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    fig.set_figheight(10)
    fig.set_figwidth(24)
    plt.subplots_adjust(
                        top = 0.94,
                        wspace = 0.05, 
                        )
    
    #fig.suptitle(("Information Plane (" + name + ", score: "+ str(MI_object.model_score) + ")"))
    fig.suptitle(("Information Plane (score: "+ str(MI_object.model_score) + ")"), fontsize=15)
    
    ax1.set_xlabel("I(X;T)")
    ax1.set_ylabel("I(T;Y)")
    ax2.set_xlabel("I(X;T)")
    ax2.set_ylabel("I(T;Y)")
    
    activations = MI_object.act_func
    cmap = plt.get_cmap('gnuplot')
    
    last_it = np.amax(list(MI_object.mi_x.keys()))
    colors = [cmap(i) for i in np.linspace(0, 1, last_it + 1)]
    color_list = color_l
    activations = MI_object.act_func
    
    
    mi_x_list = []
    mi_y_list = []
    label_count = 0
    sp_label_count = 0
    act_count = 0
    for key in MI_object.mi_x.keys():
        #epochview
        if act_count < len(activations):
            mi_x_list.append(MI_object.mi_x[key])
            mi_y_list.append(MI_object.mi_y[key])
            act_count += 1
            if act_count == len(activations):
                c = colors[key[0]]
                ax1.plot(mi_x_list, mi_y_list, marker="o",
                        markersize=9, linewidth=0.2, color=c)
                act_count = 0
                mi_x_list = []
                mi_y_list = []
        
        # layerview
        if key[0] == 0:
            if sp_label_count == 0:
                ax2.scatter(MI_object.mi_x[key], MI_object.mi_y[key],
                       color='black', linewidth=5, label='start')
                sp_label_count += 1
            else:
                ax2.scatter(MI_object.mi_x[key], MI_object.mi_y[key],
                       color='black', linewidth=5)
        elif key[0] == list(MI_object.mi_x.keys())[-1][0]:
            if sp_label_count == 1:
                ax2.scatter(MI_object.mi_x[key], MI_object.mi_y[key],
                       color='black', linewidth=5, marker="v", label='end')
                sp_label_count += 1
            else:
                ax2.scatter(MI_object.mi_x[key], MI_object.mi_y[key],
                       color='black', linewidth=5, marker="v")
        else:
            if label_count < len(activations):
                ax2.scatter(MI_object.mi_x[key],
                       MI_object.mi_y[key], color=color_list[key[1]], label="l"+str(key[1]+1)+ " " +activations[key[1]])
                label_count += 1
            else:
                ax2.scatter(MI_object.mi_x[key],
                       MI_object.mi_y[key], color=color_list[key[1]])
    
        ax2.legend()
    
    ax1.set_xlim(left=0, right=None)
    ax1.set_ylim(bottom=0, top=None)
    ax2.set_xlim(left=0, right = None)
    ax2.set_ylim(bottom=0, top=None)
    ax1.set_xbound(lower=-0.05)
    ax2.set_xbound(lower=-0.05)
    ax1.set_ybound(lower=-0.05)
    ax2.set_ybound(lower=-0.05)
    remove_neg_ticks(ax1, "x")
    remove_neg_ticks(ax1, "y")
    remove_neg_ticks(ax2, "x")
    remove_neg_ticks(ax2, "y")
    
    
    #fig.tight_layout()
    plt.tight_layout()
    if save_flag == True:
        if not os.path.exists("Results/Plots/Combinationview/"):
                try:
                    os.makedirs("Results/Plots/Combinationview/")
                except OSError as error:
                    if error.errno != errno.EEXIST:
                        raise
        plt.savefig("Results/Plots/Combinationview/" + name + "_combinationview.png")
        
    if show_flag == True:
        plt.show()
    else:
        plt.close()
    

def plot_info_plane(MI_object, name, separate_flag, color_l, show_flag, save_flag):
    """
    starts information plane plotting and creates plots for epoch, layer and separated view 
    MI_object: mutual information object
    name: name of the network
    color_l: list of colours for different layers
    show_flag:flag that decides if plot should be displayed
    save_flag: flag that decides if plot should be saved
    """
    fontsize = "15"
    params = {  #'figure.autolayout':True,
                'legend.fontsize': "12",
                'axes.labelsize': fontsize,
                'axes.titlesize': fontsize,
                'xtick.labelsize':fontsize,
                'ytick.labelsize':fontsize}
    plt.rcParams.update(params)
    
    plot_info_plane_layer_view(MI_object, name, color_l, show_flag, save_flag)
    plot_info_plane_epoch_view(MI_object, name, show_flag, save_flag)
    plot_info_plane_combination_view(MI_object, name, color_l, show_flag, save_flag)
    if separate_flag == True:
        plot_separate_info_plane_layer_view(MI_object, name, color_l, show_flag, save_flag)