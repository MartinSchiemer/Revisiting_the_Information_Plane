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

---

The goal of this framework is the creation of the Information Plane as introduced by the paper "Opening the Black Box of Deep Neural Networks via Information" by Ravid Shwartz-Ziv and
Naftali Tishby (2017)
To do so, it allows for easy data collection using a Keras callback function.
With the recorded data a simple function call calculates mutual information with a prior specified
estimator

Estimator options:<br/>
    1. Binning<br/>
    2. EDGE<br/>
    3. KDE upper bound<br/>
    4. KDE lower bound<br/>
    5. KSG discrete (-data)<br/>
    6. KSG mixture<br/>
    7. KSG continous<br/>
    

Example Jupyter Notebook files are provided illustrating the process

To run the experiments for the two datastes simply run:
python3 data_collection.py and python3 MNISTdata_collection.py or the relevant notebook.

---

## General Usage of the FrameWork

Steps for data collection:
1. define outputobject to store values
    outputs = classes.Outputs()

2. define model:
e.g.:<br/>

         model = Sequential()<br/>
         model.add(InputLayer((x_train.shape[1],)))<br/>
         model.add(Dense(10))<br/>
         model.add(Activation("tanh"))<br/>
         model.add(Dense(7))<br/>
         model.add(Activation("tanh"))<br/>
         model.add(Flatten())<br/>
         model.add(Dense(y_train.shape[1]))<br/>
         model.add(Activation("softmax"))

   or load one of the predefined ones (archhitecture is predefined name)
   e. g. 
   
    model, architecture = model_selection.select_model(model_nr, nr_of_epochs, dataset_name, X_train.shape, y_train)

   available model_nr: <br/>
   model 1 = model with leading ReLU<br/>
   model 2 = model with leading TanH<br/>
   model 3 = full ReLU<br/>
   model 4 = full TanH<br/>
        .
        .
        .
            
3. define callback function:<br/>
     
         output_recording = LambdaCallback(on_epoch_end=lambda epoch,
                                   logs: Callbacks.record_activations(outputs, model, epoch,
                                                               X_train, X_test, y_test, batch,
                                                               record_all_flag, rec_test_flag,
                                                               specific_records=[nr_of_epochs-1]))<br/>
  Parameters:  ⋅⋅* outputs: output object<br/>
               ⋅⋅* model: keras model<br/>
               ⋅⋅* epoch: current epoch index (automatically put it dont change)<br/>
               ⋅⋅* data_input: traning input data<br/>
               ⋅⋅* x_test: test data input<br/>
               ⋅⋅* y_test: test data output<br/>
               ⋅⋅* batchsize: batch size for training<br/>
               ⋅⋅* record_all_flag: flag that decides if all epochs are recorded or
                                 reduced amount (full increases comput. complexity significantly)<br/>
               ⋅⋅* rec_test_flag: flag that decides if intermediate test scores are recorded<br/>
               ⋅⋅* specific_records: list of specific epochs that shoould be recorded<br/>   
                
4. define optimzer and compile model like in normal Keras

5. fit model and add the callback function. also safe history:<br/>
    e.g. history = model.fit(X_train, y_train, epochs=nr_of_epochs, batch_size=batch,
                        validation_split=0.2, callbacks = [output_recording])
                                                               
6. get model score and add to output object
   (and history if you want to plot it using plotting.plot_history)<br/>
    e.g.<br/>
    score = model.evaluate(X_test, y_test, verbose=0)[1]<br/>
    outputs.model_score = score<br/>                      
                                                                                    
7. potentially plot loss functions and test development:    <br/>
e.g.  plotting.plot_history(history, network_name, show_flag, save_plot_flag)
                plotting.plot_test_development(outputs.int_model_score, common_name, show_flag,
                                               save_plot_flag)
          Parameters: 
              show_flag: show plot or not
              save_plot_flag: safe plot to directory or not
                                                               
8. choose estimator and define flag for it:<br/>
    est_type_flag = 1 (Binning)<br/>
                    2 (EDGE)<br/>
                    3 (KDE upper bound)<br/>
                    4 (KDE lower bound)<br/>
                    5 (KSG discrete (-data))<br/>
                    6 (KSG mixture)<br/>
                    7 (KSG continous)<br/>
                    
9. if binning was chosen define whether bin amount or size is used and size or amount of bins:<br/>
    bin_size_or_nr=True (True is size)
    bin_amount = 0.001
    
10. define colour list for plots:<br/>
     e.g. color_list = ["red", "blue", "green", "orange", "purple",
                        "brown", "pink", "teal", "goldenrod"]
   
11. decide is separated information plane should be plotted as well (single plot for each layer):<br/>
     separate_flag = False
                                                               
12. start information plane function and decide whether to calculate in parallel or sequential:<br/>
    info_plane.create_infoplane(name, X_train, y_train, outputs,
                                est_type_flag, color_list, bin_amount,
                                bin_size_or_nr, show_flag,
                                separate_flag,
                                save_flag, par_flag=parallel)
