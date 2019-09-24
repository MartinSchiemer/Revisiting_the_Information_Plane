"""
Author: Martin Schiemer
records data for "Opening the Blackbox" dataset and creates information planes with
different estimators

To change parameters change values in lines 40-55 and line 65
"""


# seeds for reproducibility
from numpy.random import seed
seed(1337)
from tensorflow import set_random_seed
set_random_seed(1337)

import numpy as np

# tensorflow properties
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers

# to implement functions at certain learning steps e.g. record weights
from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras.callbacks import EarlyStopping

# imports from other files
import Callbacks
import binning
import model_selection
import data_selection
import classes
import util
import plotting
import info_plane

def collect():
    
    name_list = ["tishby",]
    nr_of_epochs = 8000
    record_all_flag=False
    rec_test_flag=True
    learning_r = [0.0004]
    save_data_flag = False
    show_flag = False
    separate_flag = False
    save_MI_and_plot_flag = True
    
    
    bin_size_or_nr = [True]
    bins = [0.01, 0.07, 0.15, 0.3]
    
    color_list = ["red", "blue", "green", "orange", "purple",
                   "brown", "pink", "teal", "goldenrod"]


    models = [1,2,4,3]

    
    for set_name in name_list:
        seed(1337)
        set_random_seed(1337)
        X_train, X_test, y_train, y_test = data_selection.select_data(set_name, shuffle=True,
                                                                     )
        
        batch_size = [256,
                      #X_train.shape[0],128,512,1
                     ]
        
        print("calculations starting for: ", set_name)
        for i in models:

            for batch in batch_size:

                seed(1337)
                set_random_seed(1337)

                # object to record parameters
                outputs = classes.Outputs()

                # define and train model
                output_recording = LambdaCallback(on_epoch_end=lambda epoch,
                                                       logs: Callbacks.record_activations(outputs,
                                                                                          model,
                                                                                          epoch,
                                                                                          X_train,
                                                                                          X_test,
                                                                                          y_test,
                                                                                          batch,
                                                                                    record_all_flag,                                                                                      rec_test_flag))

                model, architecture = model_selection.select_model(i, nr_of_epochs,
                                                                   set_name, X_train.shape,
                                                                   y_train)



                adam = optimizers.Adam(lr=learning_r)
                model.compile(loss="categorical_crossentropy",
                              optimizer=adam,
                              metrics=["accuracy"])

                
                history = model.fit(X_train, y_train, epochs=nr_of_epochs, batch_size=batch,
                                    validation_split=0.2, callbacks = [output_recording])
                # final model score
                score = model.evaluate(X_test, y_test, verbose=0)
                score = score[1]

                # save data
                common_name = architecture + "_lr_" + str(learning_r) + "_batchsize_" + str(batch)
                if "mnist" in set_name:
                    common_name = str(samples) + str(nrs) + common_name

                aname = common_name + "_activations"
                outputs.model_score = score
                if save_data_flag == True:
                    util.save(outputs, aname)

                hname = common_name + "_history"
                h_obj = history.history
                h_obj["model_score"] = score
                if save_data_flag == True:
                    util.save(h_obj, hname)


                plotting.plot_history(h_obj, common_name, show_flag, save_MI_and_plot_flag)
                
                if rec_test_flag == True:
                    plotting.plot_test_development(outputs.int_model_score,
                                               common_name, show_flag, save_MI_and_plot_flag)    

                
                # compute binning MI
                for flag in bin_size_or_nr:
                    for nr_of_bins in bins:
                        if flag == True and nr_of_bins > 1:
                            continue
                        if flag == False and nr_of_bins < 1:
                            continue
                        seed(1337)
                        set_random_seed(1337)
                        est_type_flag = 1
                        info_plane.create_infoplane(common_name, X_train,
                                                    y_train, outputs,
                                                    est_type_flag, color_list, nr_of_bins,
                                                    flag, show_flag, separate_flag,
                                                    save_MI_and_plot_flag, par_flag=False)

                        
 
                
                
if __name__ == "__main__":
    collect()                
                
                
                
                