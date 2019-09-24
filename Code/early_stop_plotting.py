"""
Author: Martin Schiemer
additional plotting functionality for ealy stopping analysis stopping
"""

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import os
import sys
import math
import plotting



def max_fix_plot_history(h_object, max_epoch_h_object, name, show_flag, save_flag):
    print("creating history plot")
    keys = [*h_object.keys()]
    last_it = len(h_object["loss"]) - 1
    
    # find max y for full epochs
    max_epoch = len(max_epoch_h_object["loss"])
    max_y_l = np.amin(max_epoch_h_object["loss"])
    max_y_vl = np.amin(max_epoch_h_object["val_loss"])
    
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
        ax1.set_title('Model Loss (' + name + " score: " +
                      str(h_object["model_score"]) + ", last epoch: " + str(last_it + 1) + ")" )
        ax1.set_ylabel('loss')
        ax1.set_xlabel('epoch')
        ax1.legend(['train', 'validation'], loc='upper left')
        
        ax1.set_xlim(left=None, right=max_epoch)
        ax1.set_ylim(bottom=0, top=max_y_l)
        ax1.set_ybound(lower=-0.05)
        plotting.remove_neg_ticks(ax1, "y")
    
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
        
        # adjust plot to full epoch boundaries
        ax1.set_xlim(left=None, right=max_epoch)
        ax1.set_ylim(bottom=0, top=max_y_l)
        ax2.set_xlim(left=None, right = max_epoch)
        ax2.set_ylim(bottom=0, top=max_y_vl)
        
        ax1.set_ybound(lower=-0.05)
        plotting.remove_neg_ticks(ax1, "y")
        ax2.set_ybound(lower=-0.05)
        plotting.remove_neg_ticks(ax2, "y")
        
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

        
def max_fix_plot_info_plane_combination_view(MI_object, max_tot_x, max_tot_y, name,
                                             color_l, show_flag, save_flag):
    fontsize = "13"
    params = {  #'figure.autolayout':True,
                'legend.fontsize': "10",
                'axes.labelsize': fontsize,
                'axes.titlesize': fontsize,
                'xtick.labelsize':fontsize,
                'ytick.labelsize':fontsize}
    plt.rcParams.update(params)
    
    print("Creating combinationview plot")
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    fig.set_figheight(10)
    fig.set_figwidth(24)
    plt.subplots_adjust(
                        top = 0.94,
                        wspace = 0.05, 
                        )
    
    fig.suptitle(("Information Plane (score: "+ str(MI_object.model_score) + ")"))
    ax1.set_xlabel("I(X,T)")
    ax1.set_ylabel("I(T,Y)")
    ax2.set_xlabel("I(X,T)")
    ax2.set_ylabel("I(T,Y)")
    
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
                       MI_object.mi_y[key], color=color_list[key[1]], label=activations[key[1]])
                label_count += 1
            else:
                ax2.scatter(MI_object.mi_x[key],
                       MI_object.mi_y[key], color=color_list[key[1]])
        
#         # makes sure that the maximum ranges are kept
        max_mi_x_list = [max_tot_x,0]
        max_mi_y_list = [0,max_tot_y]
        ax1.plot(max_mi_x_list, max_mi_y_list, marker="o", linewidth=0.2, color="white", alpha=1)
        ax2.scatter(max_tot_x, 0, linewidth=5, color="white", alpha=1)
        ax2.scatter(0, max_tot_y, linewidth=5, color="white", alpha=1)
        
        ax2.legend()
           
    
    ax1.set_xlim(left=0, right=None)
    ax1.set_ylim(bottom=0, top=None)
    ax2.set_xlim(left=0, right = None)
    ax2.set_ylim(bottom=0, top=None)
    ax1.set_xbound(lower=-0.05)
    ax2.set_xbound(lower=-0.05)
    ax1.set_ybound(lower=-0.05)
    ax2.set_ybound(lower=-0.05)
    plotting.remove_neg_ticks(ax1, "x")
    plotting.remove_neg_ticks(ax1, "y")
    plotting.remove_neg_ticks(ax2, "x")
    plotting.remove_neg_ticks(ax2, "y")
    
    
#     ax1.set_xlim(left=0, right=None)
#     ax1.set_ylim(bottom=0, top=None)
#     ax1.set_xbound(lower=-0.05)
#     ax1.set_ybound(lower=-0.05)
    
    
#     ax2.set_xlim(left=0, right=None)
#     ax2.set_ylim(bottom=0, top=None)
#     ax2.set_xbound(lower=-0.05)
#     ax2.set_ybound(lower=-0.05)
#     plotting.remove_neg_ticks(ax1, "x")
#     plotting.remove_neg_ticks(ax1, "y")
#     plotting.remove_neg_ticks(ax2, "x")
#     plotting.remove_neg_ticks(ax2, "y")
    
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
    