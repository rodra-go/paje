import numpy as np
from tensorflow.python.lib.io import file_io
from sklearn.externals import joblib
import keras

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, input_columns, total_steps, input_sequence_length, target_sequence_length, data_path, filename_pattern, run_mode='train', start_on_batch = 0):
        'Initialization'
        self.total_steps = total_steps
        self.input_sequence_length = input_sequence_length
        self.target_sequence_length = target_sequence_length
        self.data_path = data_path
        self.run_mode = run_mode
        self.input_columns = input_columns
        self.filename_pattern = filename_pattern
        self.on_epoch_end()
        self.start_on_batch = start_on_batch
    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(self.total_steps)

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        
        if self.start_on_batch <= 0:
            step = index + 1
        else:
            step = index + self.start_on_batch

        # Generate data
        [encoder_input, decoder_input], decoder_output = self.__data_generation(step)

        return ([encoder_input, decoder_input], decoder_output)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.total_steps)
    
    def load_scaler(self, column_name):
        'Loads the scaler for the column'
        scaler_filename = 'scaler_{}.save'.format(column_name)
        with file_io.FileIO(self.data_path + scaler_filename, mode ='rb') as f:
            self.scaler = joblib.load(f)
        
    def scaler_transform(self, x):
        return self.scaler.transform(x.reshape(-1,1))

    def __data_generation(self, step):
        'Generates data containing batch_size samples'
        
        signals = []
        for column_number in range(len(self.input_columns)):
            column = self.input_columns[column_number]
            if column != 'datetime':
                self.load_scaler(column)
            filename = self.data_path + self.filename_pattern + '-' + column + '-' + self.run_mode + "-{}.csv".format("{:010d}".format(step))
            with file_io.FileIO(filename, mode ='r') as f:
                if column != 'datetime':
                    column_signals = np.genfromtxt(f, delimiter=",")
                    scaled_signals = np.array([self.scaler_transform(sequence) for sequence in column_signals])
                    signals.append(scaled_signals)
                else:
                    column_signals = np.genfromtxt(f, delimiter=",", dtype=str)
                    datetime_signals = np.array([sequence for sequence in column_signals])
                    datetime_signals = np.expand_dims(datetime_signals, axis=2)
                    signals.append(datetime_signals)

        signals = np.concatenate(signals, axis=2)
        
        encoder_input = signals[:, :self.input_sequence_length, :] # All the features
        decoder_output = signals[:, self.input_sequence_length:, -1:] # Only the feature to be predicted
        
        if len(self.input_columns) == 1:
            decoder_input = np.zeros((decoder_output.shape[0], decoder_output.shape[1], 1))
        else:
            decoder_input = signals[:, self.input_sequence_length:, :-1] # All the features but the one to be predicted
        
        # The output of the generator must be ([encoder_input, decoder_input], [decoder_output])
        return ([encoder_input, decoder_input], decoder_output)