import pytest
import pyux
import keras
import json
import os


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
        str_diff = '\n'.join([str(x) for x in diff])
        raise Exception("API change detected ! \n " + str_diff)


if __name__ == '__main__':
    pytest.main([__file__])
