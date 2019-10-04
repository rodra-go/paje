# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Example implementation of code to run on the Cloud ML service.
"""

import traceback
import argparse
import json
import os
import ast
from . import model

##Running the app
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Input Arguments
    parser.add_argument(
      '--train_data_path',
      help='GCS or local path to training data',
      required=True
    )
    parser.add_argument(
        '--eval_data_path',
        help='GCS or local path to evaluation data',
        required=True
    )
    parser.add_argument(
        '--epochs',
        help="""\
        This model works with fixed length sequences. 1-(N-1) are inputs, last is output
        """,
        required=True
    )
    parser.add_argument(
        '--output_dir',
        help='GCS location to write checkpoints and export models',
        required=True
    )
    parser.add_argument(
        '--logs_dir',
        help='GCS location to write save tensorboard logs',
        required=True
    )
    parser.add_argument(
        '--steps_per_epoch',
        help="""\
        Steps to run the training job for each epoch. A step is one batch-size,\
        """,
        required=True
    )
    parser.add_argument(
        '--validation_steps',
        help="""\
        Steps to run the validation,\
        """,
        required=True
    )
    parser.add_argument(
        '--input_columns',
        help="""\
        Columns to be loaded as inputs with the one to be predicted on the last position,\
        """,
        nargs='+',
        required=True
    )
    parser.add_argument(
        '--layers',
        help="""\
        Number of hidden neuros in each layer of the encoder and decoder,\
        """,
        nargs='+',
        required=True
    )
    parser.add_argument(
        '--filename_pattern',
        help="""\
        Steps to run the validation,\
        """,
        required=True
    )
    parser.add_argument(
        '--model_name',
        help="""\
        Name to be given to the model,\
        """,
        required=True
    )
    parser.add_argument(
        '--rnn_cell_type',
        help="""\
        Name to be given to the model,\
        """,
        required=True
    )
    parser.add_argument(
        '--input_dropout',
        help="""\
        Dropout value on the input layer,\
        """,
        required=True
    )
    parser.add_argument(
        '--recurrent_dropout',
        help="""\
        Dropout value for the recurrent layer,\
        """,
        required=True
    )
    parser.add_argument(
        '--train_batch_size',
        help='Batch size for training steps',
        type=int,
        default=200
    )
    parser.add_argument(
        '--learning_rate',
        help='Initial learning rate for training',
        type=float,
        default=0.01
    )
    parser.add_argument(
        '--input_sequence_length',
        help="""\
        This model works with sequence inputs of a fixed length.
        """,
        type=int,
        default=24
    )
    parser.add_argument(
        '--target_sequence_length',
        help="""\
        This model predicts sequence outputs of a fixed lenght.
        """,
        type=int,
        default=24
    )
    parser.add_argument(
        '--num_input_features',
        help="""\
        Number of series given as inputs.
        """,
        type=int,
        default=1
    )
    parser.add_argument(
        '--decay',
        help="""\
        Learning rate decay.
        """,
        type=float,
        default=0
    )
    parser.add_argument(
        '--job-dir',
        help='this model ignores this field, but it is required by gcloud',
        default='junk'
    )

    args = parser.parse_args()
    hparams = args.__dict__

    # unused args provided by service
    hparams.pop('job_dir', None)
    hparams.pop('job-dir', None)
    hparams['layers'] = [ int(x) for x in hparams['layers'] ]

    output_dir = hparams.pop('output_dir')

    # Append trial_id to path if we are doing hptuning
    # This code can be removed if you are not using hyperparameter tuning
    output_dir = os.path.join(
        output_dir,
        json.loads(
            os.environ.get('TF_CONFIG', '{}')
        ).get('task', {}).get('trial', '')
    )
    
    model.train_and_evaluate(output_dir, hparams)