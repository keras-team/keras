'''Canonical example of using attention for sequence to sequence problems.

This script demonstrates how to use an RNNAttentionCell to implement attention
mechanisms. In the example, the model only have to learn to filter the attended
input to obtain the target. Basically it has to learn to "parse" the attended
input sequence and output only relevant parts.

# Explanation of data:

One sample of input data consists of a sequence of one-hot-vectors separated
by randomly added "extra" zero-vectors:

    0 0 0 0 1 0 0 0 0 0
    1 0 0 1 0 0 0 0 0 1
    0 0 0 0 0 0 1 0 0 0
    0 0 1 0 0 0 0 0 0 0
    ^         ^
    |         |
    |         extra zero-vector
    one-hot vector

The goal is to retrieve the one-hot-vector sequence _without_ the extra zeros:

    0 0 0 1 0 0
    1 0 1 0 0 1
    0 0 0 0 1 0
    0 1 0 0 0 0

# Summary of the algorithm

The task is carried out by letting a Mixture Of Gaussian 1D attention mechanism
attend to the input sequence (with the extra zeros) and select what information
should be passed to the wrapped LSTM cell.

# Attention vs. Encoder-Decoder approach
This is good example where attention mechanisms are suitable. In this case
attention clearly outperforms e.g. encoder-decoder approaches.
TODO add this comparison to the script
TODO add comparison heads=1 vs heads=2 (later converges faster)
'''

from __future__ import division, print_function

import random

import numpy as np

from keras import Input
from keras.engine import Model
from keras.layers import Dense, TimeDistributed, LSTMCell, RNN

from keras.layers.attention import MixtureOfGaussian1DAttention


def get_training_data(n_samples,
                      n_labels,
                      n_timesteps_attended,
                      n_timesteps_labels):
    labels = np.random.randint(
        n_labels,
        size=(n_samples, n_timesteps_labels)
    )
    attended_time_idx = range(n_timesteps_attended)
    label_time_idx = range(1, n_timesteps_labels + 1)

    labels_one_hot = np.zeros((n_samples, n_timesteps_labels + 1, n_labels))
    attended = np.zeros((n_samples, n_timesteps_attended, n_labels))
    for i in range(n_samples):
        labels_one_hot[i][label_time_idx, labels[i]] = 1
        positions = sorted(random.sample(attended_time_idx, n_timesteps_labels))
        attended[i][positions, labels[i]] = 1

    return labels_one_hot, attended


n_samples = 10000
n_timesteps_labels = 10
n_timesteps_attended = 30
n_labels = 4

input_labels = Input((n_timesteps_labels, n_labels))
attended = Input((n_timesteps_attended, n_labels))

cell = MixtureOfGaussian1DAttention(LSTMCell(64), components=3, heads=2)
attention_lstm = RNN(cell, return_sequences=True)

attention_lstm_output = attention_lstm(input_labels, constants=attended)
output_layer = TimeDistributed(Dense(n_labels, activation='softmax'))
output = output_layer(attention_lstm_output)

model = Model(inputs=[input_labels, attended], outputs=output)

labels_data, attended_data = get_training_data(n_samples,
                                               n_labels,
                                               n_timesteps_attended,
                                               n_timesteps_labels)
input_labels_data = labels_data[:, :-1, :]
target_labels_data = labels_data[:, 1:, :]

model.compile(optimizer='Adam', loss='categorical_crossentropy')
model.fit(x=[input_labels_data, attended_data], y=target_labels_data, epochs=5)
output_data = model.predict([input_labels_data, attended_data])
