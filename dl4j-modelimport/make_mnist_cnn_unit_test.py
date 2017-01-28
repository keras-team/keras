'''
Create a simple ConvNet for testing Keras model import. Run
Keras mnist_cnn.py example and then save that model and its
outputs to disk.
'''
from __future__ import print_function

import imp
from util import save_model_details, save_model_output

SCRIPT_PATH = '../examples/mnist_cnn.py'
PREFIX = 'mnist_cnn'
OUT_DIR = '.'

print('Entering Keras script')
example = imp.load_source('example', SCRIPT_PATH)

print('Saving model details')
save_model_details(example.model, prefix=PREFIX, out_dir=OUT_DIR)

print('Saving model outputs')
save_model_output(example.model, example.X_test, example.Y_test, prefix=PREFIX, out_dir=OUT_DIR)

print('DONE!')
