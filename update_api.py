import pyux
import keras
import json


import keras.backend.tensorflow_backend
import keras.backend.theano_backend
import keras.backend.cntk_backend
import keras.backend.numpy_backend
import keras.utils.test_utils

sign = pyux.sign(keras)

with open('api.json', 'w') as f:
    json.dump(sign, f)
