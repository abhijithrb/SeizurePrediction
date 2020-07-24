# SeizurePrediction

This project uses a CNN + LSTM architecture to predict seizure from EEG data. It classifies the data as preictal(label = 1) or interictal (label = 0 ).

Ths project has 2 parts:

1) Preprocessing:
- The preprocessing script denoises the EEG signals using wavelet transform, reduces sampling freq and splits the 10 minute segment into 15 time sequences.
- This part of the project was written in MATLAB.
- the script is located under source/Preprocessing/Preprocess_data.m.

2) CNN + LSTM:
- Once the preprocessing is done, a CNN + LSTM model is trained using this data.

- the architecture is shown in the figure below:

![alt text](https://github.com/abhijithrb/SeizurePrediction/blob/master/img/net_arch.png)

- The source/DataGenerator.py script is a custom class to loads data into memory in batches instead of loading the entire dataset at once. Please see the comments in the script for more information about the class.

NOTE: To install the python libraries used:
  - Download this repository. 
  - Instal the dependencies using the command: pip3 install -r requirements.txt (this assumes python3 and pip are already installed)
  
- The source/seizure_prediction.py either trains the model or evaluates the model on the public set
  - to train the model, run the command `python3 seizure_prediction.py --mode train`
  - to test the model, run the command `python3 seizure_prediction.py --mode test`
  
  The following screenshot shows the program running for train mode:
  
  ![alt text](https://github.com/abhijithrb/SeizurePrediction/blob/master/img/Screenshot.png)

NOTE: 
- Due to size constraints, I am unable to upload the model. 
- Also due to size constraints, I have only uploaded 4 data files for each class in the training and testing folders.
