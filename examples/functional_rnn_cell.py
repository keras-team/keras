from __future__ import division, print_function

import numpy as np

from keras import Input
from keras.models import Model
from keras.layers import add, concatenate, multiply, Dense, Activation
from keras.layers.recurrent import FunctionalRNNCell, RNN, AttentionRNN

units = 32
input_size = 5

# Use functional API to define RNN Cell transformation (in this case
# simple vanilla RNN) for a single time step:
x = Input((input_size,))
h_in = Input((units,))
h_ = add([Dense(units)(x), Dense(units, use_bias=False)(h_in)])
h_out = Activation('tanh')(h_)
cell = FunctionalRNNCell(inputs=x,
                         outputs=h_out,
                         input_states=h_in,
                         output_states=h_out)

# Inject cell in RNN and apply to input sequence
x_sequence = Input((None, input_size))
rnn = RNN(cell)
y = rnn(x_sequence)

# Modify the cell to make use of attention (condition transformation on
# "external" constants such as an image or another sequence):
attended_shape = (10,)
attended = Input(attended_shape)
attention_density = Dense(attended_shape[0], activation='softmax')(
    concatenate([x, h_in]))
attention = multiply([attention_density, attended])
h2_ = add([h_, Dense(units)(attention)])
h2_out = Activation('tanh')(h2_)
attention_cell = FunctionalRNNCell(inputs=x,
                                   outputs=h2_out,
                                   input_states=h_in,
                                   output_states=h2_out,
                                   attended=attended)

# Pass the attentive cell to the AttentionRNN. Note that shape of attended is
# same as in cell (no time dimension added)
attention_rnn = AttentionRNN(attention_cell)
y2 = attention_rnn(x_sequence, attended=attended)

# Apply it on some (mock) data
attention_model = Model([x_sequence, attended], y2)
x_sequence_arr = np.random.randn(3, 5, input_size)
attended_arr = np.random.randn(3, attended_shape[0])
y2_arr = attention_model.predict([x_sequence_arr, attended_arr])
