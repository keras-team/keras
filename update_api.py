import pyux
import keras
import json

sign = pyux.sign(keras)

with open('api.json', 'w') as f:
    json.dump(sign, f)
