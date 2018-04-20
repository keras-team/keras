""" Main entry point for running benchmarks with different Keras backends.
Credit:
Script modified from TensorFlow Benchmark repo:
https://github.com/tensorflow/benchmarks/blob/keras-benchmarks/scripts/keras_benchmarks/run_benchmark.py
"""

import argparse
import json
import keras
import sys
from models import model_config

if keras.backend.backend() == "tensorflow":
    import tensorflow as tf
if keras.backend.backend() == "mxnet":
    import mxnet

parser = argparse.ArgumentParser()
parser.add_argument('--pwd',
                    help='The benchmark scripts dir')
parser.add_argument('--inference',
                    help='Benchmark inference only, use True or False')
parser.add_argument('--mode',
                    help='The benchmark can be run on cpu, gpu and multiple gpus.')
parser.add_argument('--model_name',
                    help='The name of the model that will be benchmarked.')
parser.add_argument('--dry_run', type=bool,
                    help='Flag to output metrics to the console instead of '
                         'uploading metrics to BigQuery. This is useful when '
                         'you are testing new models and do not want data '
                         'corruption.')

args = parser.parse_args()

inference = False
if args.inference:
    if args.inference not in ['True', 'False']:
        print('inference only accept True or False as parameter')
        sys.exit()

    if args.inference == 'True':
        inference = True

# Load the json config file for the requested mode.
config_file = open(args.pwd + "/config.json", 'r')
config_contents = config_file.read()
config = json.loads(config_contents)[args.mode]


def get_backend_version():
    if keras.backend.backend() == "tensorflow":
        return tf.__version__
    if keras.backend.backend() == "mxnet":
        return mxnet.__version__
    return "undefined"


model = model_config.get_model_config(args.model_name)

use_dataset_tensors = False
model.run_benchmark(gpus=config['gpus'], inference=inference, use_dataset_tensors=use_dataset_tensors)
if args.dry_run:
    print("Model :total_time", model.test_name, model.total_time)
