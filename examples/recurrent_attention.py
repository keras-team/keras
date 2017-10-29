from __future__ import division, print_function

import random

import numpy as np

from keras import Input
from keras.engine import Model
from keras.layers import Dense, TimeDistributed, LSTMCell, RNN

from keras.layers.attention import MixtureOfGaussian1DAttention

# canonical example of attention for alignment
# in this example the model should learn to "parse" through and attended
# sequence and output only relevant parts


# TODO:
# - add proper docs
# - same format as other examples
# - add encoder-decoder version for comparison of parameters efficiency
# - compare use_delta=True/False (converges faster with True)

def get_training_data(
    n_samples,
    n_labels,
    n_timesteps_attended,
    n_timesteps_labels,
):
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

cell = MixtureOfGaussian1DAttention(LSTMCell(64), n_components=3)
attention_lstm = RNN(cell, return_sequences=True)

attention_lstm_output = attention_lstm(input_labels, constants=attended)
output_layer = TimeDistributed(Dense(n_labels, activation='softmax'))
output = output_layer(attention_lstm_output)

model = Model(
    inputs=[input_labels, attended],
    outputs=output
)

labels_data, attended_data = get_training_data(
    n_samples,
    n_labels,
    n_timesteps_attended,
    n_timesteps_labels
)
input_labels_data = labels_data[:, :-1, :]
target_labels_data = labels_data[:, 1:, :]

model.compile(optimizer='Adam', loss='categorical_crossentropy')
model.fit(
    x=[input_labels_data, attended_data],
    y=target_labels_data,
    nb_epoch=5
)
output_data = model.predict([input_labels_data, attended_data])
