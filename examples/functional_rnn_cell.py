from __future__ import division, print_function

import numpy as np

from keras import Input
from keras.layers import add, Dense, Activation, FunctionalRNNCell, RNN, \
    concatenate, multiply, Model

units = 32
input_size = 5
x = Input((input_size,))
h_tm1 = Input((units,))
h_ = add([Dense(units)(x), Dense(units, use_bias=False)(h_tm1)])
h = Activation('tanh')(h_)

# Create the cell:

cell = FunctionalRNNCell(
    inputs=x, outputs=h, input_states=h_tm1, output_states=h)

x_sequence = Input((None, input_size))
rnn = RNN(cell)
y = rnn(x_sequence)

# Now we can modify the cell to make use of "external" constants:
constant_shape = (10,)
c = Input(constant_shape)
density = Dense(constant_shape[0], activation='softmax')(
    concatenate([x, h_tm1]))
attention = multiply([density, c])
h2_ = add([h, Dense(units)(attention)])
h2 = Activation('tanh')(h2_)

attention_cell = FunctionalRNNCell(
    inputs=x, outputs=h2, input_states=h_tm1, output_states=h2, constants=c)

attention_rnn = RNN(attention_cell)
y2 = attention_rnn(x_sequence, constants=c)
# Note that shape of c is same as in cell (no time dimension added)

attention_model = Model([x_sequence, c], y2)

x_sequence_arr = np.random.randn(3, 5, input_size)
c_arr = np.random.randn(3, constant_shape[0])
y2_arr = attention_model.predict([x_sequence_arr, c_arr])
