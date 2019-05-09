"""
    This script contains a custom data generator class that tf.keras calls in its fit function.
    Instead of reading the entire data to memory, it loads a batch of data, trains on the batch
    and loads the next batch till one epoch is complete

    ---
    Part of Seizure Prediction Project for the
    final project of ECGR 6119 Applied AI
    Abhijith Bagepalli
    UNC Charlotte
    May '19
"""
import numpy as np
import h5py
import tensorflow as tf


class DataGenerator(tf.keras.utils.Sequence):
    """
        Data generator class which loads a batch of data into memory instead of loading the entire
        data at once.
        This class is called internally by fit_generator function
    """

    def __init__(self, list_ids, labels, batch_size = 16, dim = (15, 16, 3200), n_channels = 1, n_classes = 2,
                 shuffle = True):
        """
            Class constructor
            ARgs:
                list_ids: list containing the file name of the data
                labels: dictionary where each element represents the class of a given filename. e.g: labels[list_ids[0]]
                        will hold the class for the file name stored in list_ids[0]
                batch_size: Size of each batch used in training
                dim: dimension of the input data -> ( # of sequences, # of rows, # of columns)
                n_channels: dimension of the matrix, since it is a 2D matrix, chanel = 1
                n_classes: Preictal and interictal

        """
        # 'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_ids
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """
            Function which calculates the number of batches per epoch
        """
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        """
            Function which Generate one batch of data
            Args:
                index: list containing the file name of the data
            Return:
                X,y: Batch of data and their corresponding labels
        """

        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_ids_temp = [self.list_IDs[k] for k in indexes]

        # Calls function to load batch of data into memory
        X, y = self.__data_generation(list_ids_temp)

        return X, y

    def on_epoch_end(self):
        """
            Updates indexes after each epoch
        """

        self.indexes = np.arange(len(self.list_IDs))
        # SHuffles the data after each epoch
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_ids_temp):
        """
            This function loads a batch of data from the mat files and feeds it to the model
            Args:
                list_ids_temp: list containing the file name of the data
            Return:
                Batch of data and their corresponding labels
        """
        # 'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty(self.batch_size, dtype = int)

        # Get data from .mat files
        for i, ID in enumerate(list_ids_temp):
            with h5py.File(ID) as file:
                _data = list(file['sequences'])

            # Convert to .mat structure to numpy array
            _npData = np.array(_data)
            _allSequences = np.transpose(_npData)

            # Reshape the numpy array to size : ( # of sequences, # of rows, # of columns, # of channels)
            X[i,] = np.reshape(_allSequences, (15, 16, 3200, 1))  # sequences

            # Store class
            y[i] = self.labels[ID]

        return X, tf.keras.utils.to_categorical(y, num_classes = self.n_classes)
