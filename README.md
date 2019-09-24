# Revisiting_the_Information_Plane

## Required Packages:

numpy 1.16.4
scipy 1.3.0
matplotlib 3.1.0
joblib 0.13.2
cvxpy 1.0.24
sklearn 0.0
jupyter notebook
tensorflow 1.13.1
pprint 0.1
seaborn 0.9.0

to install these packages simply run:
pip install PACKAGENAME


The goal of this framework is the creation of the Information Plane as introduced by the paper "Opening the Black Box of Deep Neural Networks via Information" by Ravid Shwartz-Ziv and
Naftali Tishby (2017)
To do so, it allows for easy data collection using a Keras callback function.
With the recorded data a simple function call calculates mutual information with a prior specified
estimator

Estimator options:
    1. Binning
    2. EDGE
    3. KDE upper bound
    4. KDE lower bound
    5. KSG discrete (-data)
    6. KSG mixture
    7. KSG continous
    

Example Jupyter Notebook files are provided illustrating the process

To run the experiments for the two datastes simply run:
python3 Tishbydata_collection.py and python3 MNISTdata_collection.py


## General Usage of the FrameWork

Steps for data collection:
1. define outputobject to store values
    outputs = classes.Outputs()

2. define model:
    e.g. model = Sequential()
         model.add(InputLayer((x_train.shape[1],)))
         model.add(Dense(10))
         model.add(Activation("tanh"))
         model.add(Dense(7))
         model.add(Activation("tanh"))
         model.add(Flatten())
         model.add(Dense(y_train.shape[1]))
         model.add(Activation("softmax"))

    or load one of the predefined ones (archhitecture is predefined name)
    e. g. model, architecture = model_selection.select_model(model_nr, nr_of_epochs,
                                                             dataset_name, X_train.shape, y_train)
          available model_nr: e.g. model 1 = model with leading ReLU
                                   model 2 = model with leading TanH
                                   model 3 = full ReLU
                                   model 4 = full TanH
                                   ...
            
3. define callback function:
    output_recording = LambdaCallback(on_epoch_end=lambda epoch,
                                   logs: Callbacks.record_activations(outputs, model, epoch,
                                                               X_train, X_test, y_test, batch,
                                                               record_all_flag, rec_test_flag,
                                                               specific_records=[nr_of_epochs-1]))
    Parameters: outputs: output object
                model: keras model
                epoch: current epoch index (automatically put it dont change)
                data_input: traning input data
                x_test: test data input
                y_test: test data output
                batchsize: batch size for training
                record_all_flag: flag that decides if all epochs are recorded or
                                 reduced amount (full increases comput. complexity significantly)
                rec_test_flag: flag that decides if intermediate test scores are recorded
                specific_records: list of specific epochs that shoould be recorded                   
                
4. define optimzer and compile model like in normal Keras

5. fit model and add the callback function. also safe history:
    e.g. history = model.fit(X_train, y_train, epochs=nr_of_epochs, batch_size=batch,
                        validation_split=0.2, callbacks = [output_recording])
                                                               
6. get model score and add to output object
   (and history if you want to plot it using plotting.plot_history)
    e.g.
    score = model.evaluate(X_test, y_test, verbose=0)[1]
    outputs.model_score = score                                                     
                                                                                    
7. potentially plot loss functions and test development:                                                       e.g.  plotting.plot_history(history, network_name, show_flag, save_plot_flag)
                plotting.plot_test_development(outputs.int_model_score, common_name, show_flag,
                                               save_plot_flag)
          Parameters: 
              show_flag: show plot or not
              save_plot_flag: safe plot to directory or not
                                                               
8. choose estimator and define flag for it:
    est_type_flag = 1 (Binning)
                    2 (EDGE)
                    3 (KDE upper bound)
                    4 (KDE lower bound)
                    5 (KSG discrete (-data))
                    6 (KSG mixture)
                    7 (KSG continous)
                    
9. if binning was chosen define whether bin amount or size is used and size or amount of bins
    bin_size_or_nr=True (True is size)
    bin_amount = 0.001
    
10. define colour list for plots:
     e.g. color_list = ["red", "blue", "green", "orange", "purple",
                        "brown", "pink", "teal", "goldenrod"]
   
11. decide is separated information plane should be plotted as well (single plot for each layer):
     separate_flag = False
                                                               
12. start information plane function and decide whether to calculate in parallel or sequential:
    info_plane.create_infoplane(name, X_train, y_train, outputs,
                                est_type_flag, color_list, bin_amount,
                                bin_size_or_nr, show_flag,
                                separate_flag,
                                save_flag, par_flag=parallel)
