import pytest
import numpy as np

from keras.layers import Dense, Dropout
from keras.engine.topology import merge, Input
from keras.engine.training import Model
from keras import backend as K

a = Input(shape=(32,), name='input_a')
b = Input(shape=(32,), name='input_b')

a_2 = Dense(16, name='dense_1')(a)
dp = Dropout(0.5, name='dropout')
b_2 = dp(b)

# Test recursion
model = Model([a, b], [a_2, b_2])

optimizer = 'rmsprop'
loss = 'mse'
loss_weights = [1., 0.5]
model.compile(optimizer, loss, metrics=[], loss_weights=loss_weights,
              sample_weight_mode=None)

input_a_np = np.random.random((10, 32))
input_b_np = np.random.random((10, 32))

output_a_np = np.random.random((10, 16))
output_b_np = np.random.random((10, 32))

out = model.train_on_batch([input_a_np, input_b_np],
                           [output_a_np, output_b_np])
out = model.train_on_batch({'input_a': input_a_np, 'input_b': input_b_np},
                           [output_a_np, output_b_np])
out = model.train_on_batch({'input_a': input_a_np, 'input_b': input_b_np},
                           {'dense_1': output_a_np, 'dropout': output_b_np})

out = model.test_on_batch([input_a_np, input_b_np],
                          [output_a_np, output_b_np])
out = model.test_on_batch({'input_a': input_a_np, 'input_b': input_b_np},
                          [output_a_np, output_b_np])
out = model.test_on_batch({'input_a': input_a_np, 'input_b': input_b_np},
                          {'dense_1': output_a_np, 'dropout': output_b_np})

out = model.predict_on_batch([input_a_np, input_b_np])
out = model.predict_on_batch({'input_a': input_a_np, 'input_b': input_b_np})

input_a_np = np.random.random((100, 32))
input_b_np = np.random.random((100, 32))

output_a_np = np.random.random((100, 16))
output_b_np = np.random.random((100, 32))

out = model.fit([input_a_np, input_b_np], [output_a_np, output_b_np])
out = model.evaluate([input_a_np, input_b_np], [output_a_np, output_b_np])
out = model.predict([input_a_np, input_b_np])


# with sample_weight

input_a_np = np.random.random((10, 32))
input_b_np = np.random.random((10, 32))

output_a_np = np.random.random((10, 16))
output_b_np = np.random.random((10, 32))

sample_weight = [None, np.random.random((10,))]
out = model.train_on_batch([input_a_np, input_b_np],
                           [output_a_np, output_b_np],
                           sample_weight=sample_weight)

out = model.test_on_batch([input_a_np, input_b_np],
                          [output_a_np, output_b_np],
                          sample_weight=sample_weight)

# test accuracy metric
model.compile(optimizer, loss, metrics=['acc'],
              sample_weight_mode=None)

out = model.train_on_batch([input_a_np, input_b_np],
                           [output_a_np, output_b_np])
assert len(out) == 3
out = model.test_on_batch([input_a_np, input_b_np],
                          [output_a_np, output_b_np])
assert len(out) == 3

# this should also work
model.compile(optimizer, loss, metrics={'dense_1': 'acc'},
              sample_weight_mode=None)

out = model.train_on_batch([input_a_np, input_b_np],
                           [output_a_np, output_b_np])
assert len(out) == 2
out = model.test_on_batch([input_a_np, input_b_np],
                          [output_a_np, output_b_np])
assert len(out) == 2

# and this as well
model.compile(optimizer, loss, metrics={'dense_1': ['acc']},
              sample_weight_mode=None)

out = model.train_on_batch([input_a_np, input_b_np],
                           [output_a_np, output_b_np])
assert len(out) == 2
out = model.test_on_batch([input_a_np, input_b_np],
                          [output_a_np, output_b_np])
assert len(out) == 2

input_a_np = np.random.random((100, 32))
input_b_np = np.random.random((100, 32))

output_a_np = np.random.random((100, 16))
output_b_np = np.random.random((100, 32))

out = model.fit([input_a_np, input_b_np], [output_a_np, output_b_np])
out = model.evaluate([input_a_np, input_b_np], [output_a_np, output_b_np])
out = model.predict([input_a_np, input_b_np])
