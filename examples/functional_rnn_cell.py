from __future__ import division, print_function

import numpy as np

from keras import Input
from keras.layers import add, Dense, Activation, FunctionalRNNCell, RNN, \
    concatenate, multiply, Model, AttentionRNN

units = 32
input_size = 5
x = Input((input_size,))
h_in = Input((units,))
h_ = add([Dense(units)(x), Dense(units, use_bias=False)(h_in)])
h_out = Activation('tanh')(h_)

# Create the cell:
cell = FunctionalRNNCell(
    inputs=x, outputs=h_out, input_states=h_in, output_states=h_out)

x_sequence = Input((None, input_size))
rnn = RNN(cell)
y = rnn(x_sequence)

# Modify the cell to make use of attention to "external" constants:
attended_shape = (10,)
attended = Input(attended_shape)
density = Dense(attended_shape[0], activation='softmax')(
    concatenate([x, h_in]))
attention = multiply([density, attended])
h2_ = add([h_out, Dense(units)(attention)])
h_out_2 = Activation('tanh')(h2_)

attention_cell = FunctionalRNNCell(
    inputs=x,
    outputs=h_out_2,
    input_states=h_in,
    output_states=h_out_2,
    attended=attended
)

attention_rnn = AttentionRNN(attention_cell)
y2 = attention_rnn(x_sequence, attended=attended)
# Note that shape of c is same as in cell (no time dimension added)

attention_model = Model([x_sequence, attended], y2)

x_sequence_arr = np.random.randn(3, 5, input_size)
attended_arr = np.random.randn(3, attended_shape[0])
y2_arr = attention_model.predict([x_sequence_arr, attended_arr])
