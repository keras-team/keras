import pytest
import pyux
import keras
import json
import os

import keras.backend.tensorflow_backend
import keras.backend.theano_backend
import keras.backend.cntk_backend
import keras.backend.numpy_backend
import keras.utils.test_utils


def test_api():
    api_file = os.path.join(os.getcwd(), 'api.json')
    with open(api_file, 'r') as f:
        previous_api = json.load(f)
    current_api = pyux.sign(keras)
    diff = pyux.diff(current_api, previous_api)

    exceptions = [
        pyux.ADDED_ARG_WITH_DEFAULT_IN_METHOD,
        pyux.ADDED_DEFAULT_IN_METHOD
    ]

    diff = list(filter(lambda c: c[0] not in exceptions, diff))
    if diff:
        raise pyux.APIChangedException(diff)


if __name__ == '__main__':
    test_api()
