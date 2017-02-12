'''
Test import of Keras IMDB fasttext model.
'''
from __future__ import print_function

import imp
import keras.backend as K
from util import save_model_details, save_model_output

SCRIPT_PATH = '../examples/imdb_fasttext.py'
PREFIX = 'imdb_fasttext'
OUT_DIR = '.'

print('Entering Keras script')
example = imp.load_source('example', SCRIPT_PATH)

print('Saving model details')
save_model_details(example.model, prefix=PREFIX, out_dir=OUT_DIR)

print('Saving model outputs')
save_model_output(example.model, example.X_test, example.Y_test, nb_examples=100, prefix=PREFIX, out_dir=OUT_DIR)

print('DONE!')
