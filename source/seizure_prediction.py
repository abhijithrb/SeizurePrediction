"""
    This script contains the code for training and testing a CNN + LSTM model on EEG data.
    This script requires the data to be preprocessed beforehand.
    If FLAGS.mode = train, the program will train the model
        - The  program expects the training data to reside in individual folders for each patient under Data/Preprocessed
        in the current working directory. See function read_files()
        - The program creates a list of filenames for all training data and creates a dictionary of labels corresponding
        to each file name. These are used by the DataGenerator class to create batches of data.
        - The training data is split into training and validation sets based on Flags.train_percentage. Default split is
        75-25
        - The network is trained and the model is saved in FLAGS.Model_Dir
        - to view Tensorboard files, enter in command window:
            tensorboard --logdir /path/to/logfiles
            Log file path is set using FLAGS.log_dir
    If FLAGS.mode = test, the program will find the accuracy on the test data
        - The test routine requires the preprocessed data and requires all the files under one folder.
        It reuires the all the test files to be under Data/Preprocessed/Testing_Data in the current working directory
        check function get_test_data() to change paths
        - The program reads the given CSV file (see function get_test_data() for paths)
        - The csv file contains the list of files that are private set and public test set. The program only fetches the
        public test data and their labels.
        - The program creates a list of filenames for all training data and creates a dictionary of labels corresponding
        to each file name. These are used by the DataGenerator class to create batches of data.
        - The program will evaluate the model on the entire test set and display the accuracy
"""
import os
import csv
import numpy as np
import tensorflow as tf
import random
from DataGenerator import DataGenerator
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.layers.normalization import BatchNormalization
from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from tensorflow.python.keras.engine.sequential import Sequential
from tensorflow.python.keras.layers import Convolution2D, MaxPooling2D, Flatten
from tensorflow.python.keras.layers import Dense, Dropout, Activation
from tensorflow.python.keras.layers.wrappers import TimeDistributed
from tensorflow.python.keras.models import Sequential, load_model

tf.app.flags.DEFINE_string('device_id', '/GPU:0',
                           'What processing unit to execute on')

tf.app.flags.DEFINE_float('train_percentage', 0.75, 'percentage of images used for training, rest for validation')

tf.app.flags.DEFINE_integer('batch_size', 8, 'Batch size')

tf.app.flags.DEFINE_integer('epochs', 150, 'Number of epochs')

tf.app.flags.DEFINE_float('learning_rate', 0.0001, 'Learning Rate for Gradient Descent Algorithm')

tf.app.flags.DEFINE_string('log_dir', '../source/Log/',
                           'Sub Directory for storing summaries')

tf.app.flags.DEFINE_string('Model_Dir',
                           '../source/Model/',
                           'Sub Directory where checkpoints are to be saved')

tf.app.flags.DEFINE_string('mode', '',
                           'train or test')

FLAGS = tf.app.flags.FLAGS


def read_files():
    """
        Function that reads files from training_dir and creates a list of the filenames and their corresponding labels
    """
    main_path = os.getcwd()
    training_dir = ['Data/Preprocessed/Pat1Train/',
                    'Data/Preprocessed/Pat3Train',
                    'Data/Preprocessed/Pat2Train']

    label_lists = {}
    file_names = []

    items_in_class1 = 0

    for idx in range(len(training_dir)):
        pat_dir = os.path.join(main_path, training_dir[idx])
        class_names = os.listdir(pat_dir)

        for i, cl in enumerate(reversed(class_names)):
            files = os.listdir(os.path.join(pat_dir, cl))
            files = [os.path.join(pat_dir, cl) + '/' + e for e in files]

            if cl == '0':
                files = random.sample(files, items_in_class1)
            else:
                items_in_class1 = len(files)

            file_names.extend(files)

            for j in range(len(files)):
                if int(cl) == 0:
                    label_lists[files[j]] = 0
                else:
                    label_lists[files[j]] = 1

    return file_names, label_lists


def get_data():
    """
        Function that stores a list of training and validation file names and their corresponding labels
    """
    _data = {}
    _labels = {}

    data, label = read_files()
    print('label: ', label)
    np.random.shuffle(data)

    idx1 = FLAGS.train_percentage

    _data['train'] = data[0: int(idx1 * len(data))]
    _data['valid'] = data[int(idx1 * len(data)):]

    return _data, label


def get_test_data():
    """
        Function that gets the public test data and the labels from the given csv file
    """
    label_lists = {}
    file_names = []

    main_path = os.getcwd()
    path_to_test_files = os.path.join(main_path, 'Data/Preprocessed/Testing_Data')
    path_to_csv = os.path.join(main_path, 'Data/contest_test_data_labels_public.csv')

    with open(path_to_csv) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            if row[2] == 'Public':
                pat_dir = os.path.join(path_to_test_files, row[0])
                file_names.append(pat_dir)
                label_lists[pat_dir] = row[1]

    return file_names, label_lists


def build_model():
    """
        Function that build the CNN + LSTM network
    """
    with tf.name_scope('CNN_LSTM'):
        model = Sequential()

        with tf.name_scope('Conv1'):
            model.add(TimeDistributed(Convolution2D(16, (5, 5), padding = 'same', strides = (2, 2)),
                                      input_shape = (15, 16, 3200, 1), name = 'Conv1'))

        model.add(BatchNormalization())
        model.add(Activation('relu'))

        with tf.name_scope('Conv2'):
            model.add(TimeDistributed(Convolution2D(32, (5, 5), padding = 'same', strides = (1, 1), name = 'Conv2')))
            model.add(Activation('relu'))

        with tf.name_scope('Pooling'):
            model.add(TimeDistributed(MaxPooling2D(pool_size = (2, 2))))

        with tf.name_scope('Conv3'):
            model.add(TimeDistributed(Convolution2D(32, (5, 5), padding = 'same', strides = (1, 1), name = 'Conv3')))
            model.add(Activation('relu'))

        with tf.name_scope('Conv4'):
            model.add(TimeDistributed(Convolution2D(32, (5, 5), padding = 'same', strides = (1, 1), name = 'Conv4')))
            model.add(Activation('relu'))

        with tf.name_scope('Pooling'):
            model.add(TimeDistributed(MaxPooling2D(pool_size = (2, 2))))

        with tf.name_scope('FC1'):
            model.add(TimeDistributed(Flatten(), name = 'FC1'))
            model.add(Activation('relu'))

            model.add(TimeDistributed(Dropout(0.25)))

        with tf.name_scope('FC2'):
            model.add(TimeDistributed(Dense(256), name = 'FC2'))
            model.add(Activation('relu'))

            model.add(TimeDistributed(Dropout(0.25)))

        with tf.name_scope('LSTM'):
            model.add(tf.keras.layers.CuDNNLSTM(64, return_sequences = False))
            model.add(Dropout(0.5))

        with tf.name_scope('OutputLayer'):
            model.add(Dense(2, activation = 'softmax'))

    with tf.name_scope('Optimizer'):
        optimizer = optimizers.adam(lr = 1e-4, decay = 1e-5)

    with tf.name_scope('Loss'):
        model.compile(loss = 'categorical_crossentropy',
                      optimizer = optimizer,
                      metrics = ['accuracy'])

    return model


def training_fn(_generator):
    """
        Function that calls the routine to  build model and train
    """
    # Required for efficient GPU usage
    config = tf.ConfigProto(allow_soft_placement = True, log_device_placement = False)
    config.gpu_options.allow_growth = True
    config.gpu_options.allocator_type = 'BFC'
    config.gpu_options.per_process_gpu_memory_fraction = 0.90

    with tf.Session(config = config) as session:
        with tf.device(FLAGS.device_id):
            session.run(tf.global_variables_initializer())

            # Build the network architecture
            model = build_model()

            # Print the network summary
            print(model.summary())

            # Callback function to stop training if validation loss doesn't improve
            early_stopper = EarlyStopping(patience = 25)

            # Callback function to tensorboard log files
            tb = TensorBoard(FLAGS.log_dir)

            # Model name and directory
            model_dir = os.path.join(FLAGS.Model_Dir, 'CNN_LSTM.h5')

            # Save the model
            checkpoint = ModelCheckpoint(model_dir, verbose = 1,
                                         save_best_only = True)
            # Train the model
            history = model.fit_generator(generator = _generator['train'],
                                          epochs = FLAGS.epochs,
                                          verbose = 1,
                                          validation_data = _generator['valid'],
                                          use_multiprocessing = True,
                                          shuffle = True,
                                          workers = 7,
                                          max_queue_size = 15,
                                          callbacks = [tb, early_stopper, checkpoint])

            # Evaluate model on validation set once training is done.
            eval_score = model.evaluate_generator(_generator['valid'],
                                                  use_multiprocessing = True,
                                                  workers = 6)

            print('Test accuracy:', eval_score[1])


def testing_fn(_generator):
    # Required for efficient GPU usage
    config = tf.ConfigProto(allow_soft_placement = True, log_device_placement = False)
    config.gpu_options.allow_growth = True
    config.gpu_options.allocator_type = 'BFC'
    config.gpu_options.per_process_gpu_memory_fraction = 0.90

    model_dir = os.path.join(FLAGS.Model_Dir, 'lrcn_1.h5')

    with tf.Session(config = config) as session:
        with tf.device(FLAGS.device_id):
            session.run(tf.global_variables_initializer())

            # load the model we saved
            model = tf.keras.models.load_model(model_dir)

            eval_score = model.evaluate_generator(_generator,
                                                  use_multiprocessing = True,
                                                  workers = 6)

            print('Test accuracy:', eval_score[1])


def train():
    """
        Function that trains the model
    """
    # Parameters for DataGrnerator constructor
    params = {'dim': (15, 16, 3200),
              'batch_size': FLAGS.batch_size,
              'n_classes': 2,
              'n_channels': 1,
              'shuffle': True}

    # Get list of filenames for all training data and a dictionary of labels corresponding to each file name
    data, labels = get_data()

    with tf.name_scope('InputBatches'):
        # Generators for training and validation data
        generator_obj = {}
        training_generator = DataGenerator(data['train'], labels, **params)
        validation_generator = DataGenerator(data['valid'], labels, **params)

        # Stores the training and validation batches in a dict
        generator_obj['train'] = training_generator
        generator_obj['valid'] = validation_generator

    # Train the model
    training_fn(generator_obj)


def inference():
    """
        Function that calls the routine to test the model
    """

    # Parameters for DataGrnerator constructor
    params = {'dim': (15, 16, 3200),
              'batch_size': 1,
              'n_classes': 2,
              'n_channels': 1,
              'shuffle': False}

    # Get list of filenames for all public test data and a dictionary of labels corresponding to each file name
    data, labels = get_test_data()

    # Generator for test data
    test_generator = DataGenerator(data, labels, **params)

    # Calls the routine to evaluate the model on the testing set
    testing_fn(test_generator)


def main():
    """
        Main function which runs either the train routine or testing routine
    """

    if FLAGS.mode == 'train':
        train()
    elif FLAGS.mode == 'test':
        inference()
    else:
        print('Incorrect Run mode. Options are train or test')


if __name__ == "__main__":
    main()
