import sys
sys.path.insert(0, './energyforecast/trainer')
import my_classes
import json
import subprocess
import numpy as np
import pandas as pd
from statistics import mean
from math import sqrt
from sklearn.externals import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib
from sklearn.metrics import mean_squared_error
from keras.models import load_model

def get_split_arr(dataset_split,batch_number):
  dataset_split = np.array(dataset_split)
  split_arr = np.floor(dataset_split*batch_number)

  if split_arr[1] == 0:
    split_arr[1] = 1
  
  split_arr[2] = batch_number - split_arr[0] - split_arr[1]
  
  if split_arr[2] == 0:
    split_arr[2] = 1
    split_arr[0] = split_arr[0] - 1 
  split_arr = split_arr.astype(int)

  if split_arr[0] == 0:
    raise Exception('The split vector produces 0 batches for train dataset!')
  elif split_arr[1] == 0:
    raise Exception('The split vector produces 0 batches for valid dataset!')
  elif split_arr[2] == 0:
    raise Exception('The split vector produces 0 batches for test dataset!')
  
  return split_arr

def add_to_dataset_list(metadata, dataset_list_path):
  try:
    with open(dataset_list_path, 'r') as f:
      dataset_list = json.load(f)

    dataset_list[metadata['dataset_name']] = metadata
    with open(dataset_list_path, 'w') as fp:
      json.dump(dataset_list, fp)
  except:
    dataset_list = {}
    dataset_list[metadata['dataset_name']] = metadata

    with open(dataset_list_path, 'w') as fp:
      json.dump(dataset_list, fp)
    
def add_to_model_list(metadata, model_list_path):
  try:
    with open(model_list_path, 'r') as f:
      model_list = json.load(f)

    model_list[metadata['model_name']] = metadata
    with open(model_list_path, 'w') as fp:
      json.dump(model_list, fp)
  except:
    model_list = {}
    model_list[metadata['model_name']] = metadata

    with open(model_list_path, 'w') as fp:
      json.dump(model_list, fp)

def fill_missing(df,periods, column_number):
  
  # Confere número de instantes com valor nulo e número total de instantes
  number_of_nan_rows = df[df[df.columns[column_number]].isnull() == True].shape[0]
  total_rows = df.shape[0]
  print('Foram encontrados dados faltantes para {} instantes na coluna {}.\n'.format(number_of_nan_rows,df.columns.values[column_number]))
  print('Porcentagem de dados faltantes: {} %.\n'.format(round(100*number_of_nan_rows/total_rows,2)))
  
  # Inicia iterações de reconstituição com limite igual a 5
  it = 1
  while ((df[df[df.columns[column_number]].isnull() == True].shape[0] != 0) and it < 5):
    shifted_df = df.shift(periods=periods)
    df = df.fillna(shifted_df)
    print('{}º iteração, ainda há {} instantes com valores faltates.'.format(it,df[df[df.columns[column_number]].isnull() == True].shape[0]))
    it += 1
  
  # Caso ainda haja valores faltantes após as 5 iterações, o sentido de substituição é invertido,
  # e portanto o valor substituto é obtido do futuro, não do passado
  print('Invertendo sentido de substituição...')
  if(df[df[df.columns[column_number]].isnull() == True].shape[0] != 0):
    while ((df[df[df.columns[column_number]].isnull() == True].shape[0] != 0) and it < 10):
      shifted_df = df.shift(periods=-periods)
      df = df.fillna(shifted_df)
      print('{}º iteração, ainda há {} instantes com valores faltates.'.format(it,df[df[df.columns[column_number]].isnull() == True].shape[0]))
      it += 1
      
  print('Processo de reconstituição de dados faltantes concluído.')
  return df

def create_windowed_dataset(df,
                            dataset_name,
                            filename_pattern, 
                            n_input, 
                            n_output, 
                            batch_size, 
                            dataset_split,
                            output_folder,
                            validation_from_past):
  
  # Define dictonary to save metadata
  metadata = {}
    
  # Function to adjust timeserie length to batch size  
  def adjust(df,n):
    return df.iloc[-(df.shape[0]//n*n):]

  # Create output folder if necessary
  import os
  try:
      os.makedirs(output_folder)
  except OSError:
      pass

  nseq = n_input + n_output
    
  df = adjust(df,nseq)

  batch_number = df.shape[0]//batch_size
  # Calculate split array for train, validation and test datasets
  split_arr = get_split_arr(dataset_split,batch_number)

  # Use training and validation datasets to calculate the scaler
  scaling_index = ( split_arr[0] + split_arr[1] ) * batch_size
  scaling_dataset = df.drop('datetime', axis=1).iloc[:scaling_index]
  
  metadata['dataset_name'] = dataset_name
  metadata['columns'] = df.columns.values.tolist()
  metadata['total_batches'] = str(batch_number)
  metadata['train_batches'] = str(split_arr[0])
  metadata['valid_batches'] = str(split_arr[1])
  metadata['test_batches'] = str(split_arr[2])

  scaler = MinMaxScaler()
  for column_number in range(df.columns.values.shape[0]):
      column_name = df.columns.values[column_number]
      timeseries = df.values[:,column_number]
      
      if column_name != 'datetime':
        scaler = MinMaxScaler()
        scaling_series = df[column_name].iloc[:scaling_index]
        scaler.fit(scaling_series.values.reshape(-1, 1))
        scaler_filename = 'scaler_{}.save'.format(column_name)
        joblib.dump(scaler, output_folder + scaler_filename)
        print('Scaler model for column {} dumped to {}'.format(column_name, output_folder + scaler_filename))
        
      in_start = 0
      file_number = 1
      if validation_from_past:
          dataset = 'valid'
          print('Started validation dataset generation for column {}'.format(column_name))
          first_split_lenght = split_arr[1]
      else:
          dataset = 'train'
          print('Started train dataset generation for column {}'.format(column_name))
          first_split_lenght = split_arr[0]
      for batch in range(batch_number):
        if batch == first_split_lenght:
          if validation_from_past:
              dataset = 'train'
          else:
              dataset = 'valid'
          print('Started {} dataset generation for column {}'.format(dataset,column_name))
          file_number = 1
        elif batch == split_arr[0] + split_arr[1]:
          dataset = 'test'
          print('Started {} dataset generation for column {}'.format(dataset,column_name))
          file_number = 1

        # Format file number with leading zeros
        file_n = "{:010d}".format(file_number)
        series_filename = output_folder + filename_pattern + '-' + column_name + '-' + dataset + '-' + file_n + '.csv'
        with open(series_filename, 'w') as series_file:
          for _ in range(batch_size):
            # define the end of the input sequence
            in_end = in_start + n_input
            out_end = in_end + n_output
            # ensure we have enough data for this instance
            if out_end < len(timeseries):
              serie = timeseries[in_start:out_end]
              serie_line = ",".join(map(str, serie))
              series_file.write(serie_line + '\n')
            # move along one time step
            in_start += 1
        file_number += 1
        
  print('\nCreation process finished for dataset {}!'.format(metadata['dataset_name']))
  print('Total batches: {}'.format(metadata['total_batches']))
  print('Train batches: {}'.format(metadata['train_batches']))
  print('Validation batches: {}'.format(metadata['valid_batches']))
  print('Test batches: {}\n'.format(metadata['test_batches']))  
  
  with open(output_folder + 'metadata_{}.json'.format(metadata['dataset_name']), 'w') as fp:
    json.dump(metadata, fp)

  return metadata

def generate_datasets(df,input_output_vector,batch_size_vector,dataset_split,dataset_list_json_path, filename_pattern, output_folder, validation_from_past):
  for input_output in input_output_vector:
    for batch_size in batch_size_vector:
      # Sets dataset_name and output_folder
      validation_sampling_type = 'vf'
      if validation_from_past:
        validation_sampling_type = 'vp'
            
      dataset_name = '{}_{}_{}x{}_{}'.format(filename_pattern,validation_sampling_type,input_output[0],input_output[1],batch_size)
      output_f = output_folder + '{}/'.format(dataset_name)
      print('Starting generation for dataset {}...\n'.format(dataset_name))
      
      # Deletes existent directory
      cmd1 = subprocess.Popen(['rm','-rf',output_f], stdout=subprocess.PIPE)
      cmd1_output, cmd1_error = cmd1.communicate()
      
      metadata = create_windowed_dataset(df = df,
                                          dataset_name=dataset_name,
                                          filename_pattern = filename_pattern,
                                          n_input = input_output[0],
                                          n_output = input_output[1], 
                                          batch_size = batch_size, 
                                          output_folder = output_f,
                                          dataset_split = dataset_split,
                                          validation_from_past = validation_from_past)
      metadata['batch_size'] = batch_size
      metadata['n_input'] = input_output[0]
      metadata['n_output'] = input_output[1]
      metadata['dataset_split'] = dataset_split
      add_to_dataset_list(metadata,dataset_list_json_path)

def test_model(dataset_path,
               dataset_list_path,
               dataset_name,
               model_list_path,
               model_name,
               n_input,
               n_output,
               filename_pattern,
               trained_models_folder,
               load_checkpoint_model,
               run_mode='test'):
    
  # Obter informações do dataset
  with open(dataset_list_path, 'r') as f:
    dataset_list = json.load(f)
    test_steps = int(dataset_list[dataset_name]['test_batches'])
    start_on_batch = 0
    if run_mode == 'train':
        start_on_batch = int(dataset_list[dataset_name]['train_batches'])
    elif run_mode == 'valid':
        if int(dataset_list[dataset_name]['valid_batches']) < test_steps:
            test_steps = int(dataset_list[dataset_name]['valid_batches'])
        start_on_batch = int(dataset_list[dataset_name]['valid_batches'])

  # Obter informações do modelo
  with open(model_list_path, 'r') as f:
    model_list = json.load(f)
    model_metadata = model_list[model_name]
    input_cols = model_metadata['input_cols']

  # Definir do modelo para teste
  if load_checkpoint_model:
    trained_model_path = trained_models_folder + 'checkpoint_' + model_name + '.h5'
  else:
    trained_model_path = trained_models_folder + model_name + '.h5'  

  # Load the model from .h5 file
  print('Loading the trained model at {}...'.format(trained_model_path))
  model = load_model(trained_model_path)
  model.summary()
  
  # Loads dataset scaler for output column, that should be the last one on INPUT_COLS
  scaler = joblib.load(dataset_path + 'scaler_{}.save'.format(input_cols[-1:][0]))
  
  # Incializando o gerador de dados
  test_data_generator = my_classes.DataGenerator(total_steps=test_steps,
                                                 input_sequence_length=int(n_input),
                                                 target_sequence_length=int(n_output),
                                                 run_mode = run_mode,
                                                 data_path=dataset_path,
                                                 input_columns=input_cols,
                                                 filename_pattern=filename_pattern,
                                                 start_on_batch=start_on_batch - test_steps)

  # Calcular o valor médio da função de custo em todo o dataset
  if run_mode == 'train':
    print('Calculating average loss for last {} batches from train dataset...'.format(test_steps))
  if run_mode == 'valid': 
    print('Calculating average loss for last {} batches from valid dataset...'.format(test_steps))
  if run_mode == 'test':
    print('Calculating average loss on test dataset...')
  test_average_loss = model.evaluate_generator(test_data_generator, 
                                               steps=int(test_steps), 
                                               workers=1, 
                                               use_multiprocessing=False, 
                                               verbose=1)
  
  # Reinicializar o gerador de dados
  test_data_generator = my_classes.DataGenerator(total_steps=1,
                                                 input_sequence_length=int(n_input),
                                                 target_sequence_length=int(n_output),
                                                 run_mode = run_mode,
                                                 data_path=dataset_path,
                                                 input_columns=input_cols,
                                                 filename_pattern=filename_pattern,
                                                 start_on_batch=start_on_batch)

  # Calcular o valor médio da função de custo no primeiro lote de treinamento
  if run_mode == 'train':
    print('Calculating average loss for last batch from train dataset...')
  if run_mode == 'valid': 
    print('Calculating average loss for last batch from valid dataset...')
  if run_mode == 'test':
    print('Calculating average loss for first batch from test dataset...')
  test_average_loss_on_first_batch = model.evaluate_generator(test_data_generator, 
                                                             steps=1, 
                                                             workers=1, 
                                                             use_multiprocessing=False, 
                                                             verbose=1)
  
  # Reinicializar o gerador de dados
  test_data_generator = my_classes.DataGenerator(total_steps=1,
                                                 input_sequence_length=int(n_input),
                                                 target_sequence_length=int(n_output),
                                                 run_mode = run_mode,
                                                 data_path=dataset_path,
                                                 input_columns=input_cols,
                                                 filename_pattern=filename_pattern,
                                                 start_on_batch=start_on_batch)

  datetime_list_generator = my_classes.DataGenerator(total_steps=1,
                                                     input_sequence_length=int(n_input),
                                                     target_sequence_length=int(n_output),
                                                     run_mode = run_mode,
                                                     data_path=dataset_path,
                                                     input_columns=['datetime'],
                                                     filename_pattern=filename_pattern,
                                                     start_on_batch=start_on_batch)

  if run_mode == 'train':
    print('Calculating loss vector for last batch from train dataset...')
  if run_mode == 'valid': 
    print('Calculating loss vector for last batch from valid dataset...')
  if run_mode == 'test':
    print('Calculating loss vector for first batch from test dataset...')
  (encoder_i, decoder_i), decoder_o = test_data_generator.__getitem__(0)
  (encoder_dt, _ ), decoder_dt = datetime_list_generator.__getitem__(0)
  predictions = model.predict([encoder_i, decoder_i])
 
  X = []
  X_dt = []
  Y = []
  Y_hat = []
  Y_dt = []
  test_loss_vector_first_batch = []
  for sample in range(int(dataset_list[dataset_name]['batch_size'])):
    x = encoder_i[sample,:,-1:]
    x_dt = encoder_dt[sample]
    y = scaler.inverse_transform(decoder_o[sample])
    y_hat = scaler.inverse_transform(predictions[sample])
    y_dt = decoder_dt[sample]
    X.append(x.reshape(-1).tolist())
    X_dt.append(x_dt.reshape(-1).tolist())
    Y.append(y.reshape(-1).tolist())
    Y_hat.append(y_hat.reshape(-1).tolist())
    Y_dt.append(y_dt.reshape(-1).tolist())
#     X.append(x.reshape(-1))
#     X_dt.append(x_dt.reshape(-1))
#     Y.append(y.reshape(-1))
#     Y_hat.append(y_hat.reshape(-1))
#     Y_dt.append(y_dt.reshape(-1))
    test_loss_vector_first_batch.append(sqrt(mean_squared_error(y,y_hat)))

  print('Test calculations done.')

  results = {}
  results['avg_loss'] = scaler.inverse_transform(sqrt(test_average_loss))[0][0]
  results['avg_loss_first_batch'] = scaler.inverse_transform(sqrt(test_average_loss_on_first_batch))[0][0]
  results['loss_vec_first_batch'] = test_loss_vector_first_batch
  results['X_first_batch'] = X
  results['X_dt_first_batch'] = X_dt
  results['Y_first_batch'] = Y
  results['Y_hat_first_batch'] = Y_hat
  results['Y_dt_first_batch'] = Y_dt

  if load_checkpoint_model:
    model_metadata['results'] = {}
    model_metadata['results']['checkpoint'] = results
  else:
    model_metadata['results'] = {}
    model_metadata['results']['last'] = results
  add_to_model_list(model_metadata,model_list_path)

  return results