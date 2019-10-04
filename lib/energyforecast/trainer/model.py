#!/usr/bin/env python
import os
import keras as keras
import tensorflow as tf
import numpy as np

from tensorflow.python.lib.io import file_io
from . import my_classes

tf.logging.set_verbosity(tf.logging.INFO)

def s2s_model(hparams,
              num_output_features = 1):
    
    regulariser = None
    dropout = float(hparams['input_dropout'])
    recurrent_dropout = float(hparams['recurrent_dropout'])
    
    print('Starting model building...')
    num_encoder_input_features = len(hparams['input_columns'])
    if num_encoder_input_features == 1:
        num_decoder_input_features = num_encoder_input_features
    else:
        num_decoder_input_features = num_encoder_input_features - 1 # The decoder input does not contain the field to be predicted 
    
    # Define an input sequence.
    encoder_inputs = keras.layers.Input(shape=(None, num_encoder_input_features), name='encoder_inputs')
    
    # Define the layers
    layers = hparams['layers']

    # Create a list of RNN Cells, these are then concatenated into a single layer
    # with the RNN layer.
    encoder_cells = []
    for hidden_neurons in layers:
        if hparams['rnn_cell_type'] == 'GRU':
            l = keras.layers.GRUCell(hidden_neurons,
                                     kernel_regularizer=regulariser,
                                     recurrent_regularizer=regulariser,
                                     bias_regularizer=regulariser,
                                     dropout=dropout,
                                     recurrent_dropout=recurrent_dropout)
            
        elif hparams['rnn_cell_type'] == 'LSTM':
            l = keras.layers.LSTMCell(hidden_neurons,
                                     kernel_regularizer=regulariser,
                                     recurrent_regularizer=regulariser,
                                     bias_regularizer=regulariser,
                                     dropout=dropout,
                                     recurrent_dropout=recurrent_dropout)
            
        encoder_cells.append(l)

    encoder = keras.layers.RNN(encoder_cells, return_state=True)

    encoder_outputs_and_states = encoder(encoder_inputs)

    # Discard encoder outputs and only keep the states.
    # The outputs are of no interest to us, the encoder's
    # job is to create a state describing the input sequence.
    encoder_states = encoder_outputs_and_states[1:]

    # The decoder input will be set to zero (see random_sine function of the utils module).
    # Do not worry about the input size being 1, I will explain that in the next cell.
    decoder_inputs = keras.layers.Input(shape=(None, num_decoder_input_features), name='decoder_input')

    decoder_cells = []
    for hidden_neurons in layers:
        if hparams['rnn_cell_type'] == 'GRU':
            l = keras.layers.GRUCell(hidden_neurons,
                                     kernel_regularizer=regulariser,
                                     recurrent_regularizer=regulariser,
                                     bias_regularizer=regulariser,
                                     dropout=dropout,
                                     recurrent_dropout=recurrent_dropout)
            
        elif hparams['rnn_cell_type'] == 'LSTM':
            l = keras.layers.LSTMCell(hidden_neurons,
                                     kernel_regularizer=regulariser,
                                     recurrent_regularizer=regulariser,
                                     bias_regularizer=regulariser,
                                     dropout=dropout,
                                     recurrent_dropout=recurrent_dropout)
            
        decoder_cells.append(l)

    decoder = keras.layers.RNN(decoder_cells, return_sequences=True, return_state=True)

    # Set the initial state of the decoder to be the ouput state of the encoder.
    # This is the fundamental part of the encoder-decoder.
    decoder_outputs_and_states = decoder(decoder_inputs, initial_state=encoder_states)

    # Only select the output of the decoder (not the states)
    decoder_outputs = decoder_outputs_and_states[0]

    # Apply a dense layer with linear activation to set output to correct dimension
    # and scale (tanh is default activation for GRU in Keras, our output sine function can be larger then 1)
    decoder_dense = keras.layers.Dense(num_output_features,
                                       activation='linear',
                                       kernel_regularizer=regulariser,
                                       bias_regularizer=regulariser)

    decoder_outputs = decoder_dense(decoder_outputs)
    
    optimiser = keras.optimizers.Adam(lr=float(hparams['learning_rate']), decay=float(hparams['decay'])) # Other possible optimiser "sgd" (Stochastic Gradient Descent)
    loss = "mse" # Other loss functions are possible, see Keras documentation.
    
    model = keras.models.Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_outputs)
    model.compile(optimizer=optimiser, loss=loss)
    print('Model Compiled')
    
    return model

# ---------------------------------------------- #
# ------------Treinamento e avaliação----------- #
# ---------------------------------------------- #
def train_and_evaluate(output_dir, hparams):
    
    print('Started training and evaluation')

    import time as time
    from keras.callbacks import TensorBoard
    from keras.callbacks import ModelCheckpoint
    
    tf.summary.FileWriterCache.clear()
    
    ts = time.gmtime()
    
    #logs_dir = output_dir + 'tensorboard_logs/'
    logs_dir = hparams['logs_dir']
    tensorboard = TensorBoard(log_dir=logs_dir+'{}'.format(hparams['model_name']))
    checkpoint_filename = 'checkpoint_{}.h5'.format(hparams['model_name'])
    model_chekpoint = ModelCheckpoint(checkpoint_filename, monitor='val_loss', mode='min', verbose=1, save_best_only=True)
            
    print('Instantiating the model...')
    model = s2s_model(hparams)

    train_data_generator = my_classes.DataGenerator(total_steps=int(hparams['steps_per_epoch']),
                                       input_sequence_length=int(hparams['input_sequence_length']),
                                       target_sequence_length=int(hparams['target_sequence_length']),
                                       run_mode = 'train',
                                       data_path=hparams['train_data_path'],
                                       input_columns=hparams['input_columns'],
                                       filename_pattern=hparams['filename_pattern'])

    valid_data_generator = my_classes.DataGenerator(total_steps=int(hparams['validation_steps']),
                                       input_sequence_length=int(hparams['input_sequence_length']),
                                       target_sequence_length=int(hparams['target_sequence_length']),
                                       run_mode = 'valid',
                                       data_path=hparams['eval_data_path'],
                                       input_columns=hparams['input_columns'],
                                       filename_pattern=hparams['filename_pattern'])

    model.summary()

    model.fit_generator(generator=train_data_generator, 
                        steps_per_epoch=int(hparams['steps_per_epoch']), 
                        epochs=int(hparams['epochs']), 
                        callbacks=[tensorboard,model_chekpoint],
                        validation_data=valid_data_generator, 
                        validation_steps=int(hparams['validation_steps']),
                        verbose=1,
                        workers=1,
                        use_multiprocessing=False)

    # Save model.h5 on to google storage
    model_filename = '{}.h5'.format(hparams['model_name'])
    model.save(model_filename)
    with file_io.FileIO(model_filename, mode='rb') as input_f, file_io.FileIO(checkpoint_filename, mode='rb') as input_mc:
        with file_io.FileIO(output_dir + 'trained_models/' + model_filename, mode='w+') as output_f, file_io.FileIO(output_dir + 'trained_models/' + checkpoint_filename, mode='w+') as output_mc:
            output_f.write(input_f.read())
            output_mc.write(input_mc.read())
    os.remove(model_filename)
    os.remove(checkpoint_filename)