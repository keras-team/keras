import os
import pytest
import sys

# Ensure we are testing the local keras
sys.path.insert(0, os.getcwd())

os.environ['KERAS_BACKEND'] = 'openvino'
print("Running tests with KERAS_BACKEND=openvino")

# Run specific tests
args = [
    'keras/src/losses/losses_test.py',
    '-k', 'Crossentropy',
    '-v'
]

sys.exit(pytest.main(args))
