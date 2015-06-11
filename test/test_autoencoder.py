from __future__ import absolute_import
from __future__ import print_function
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import DenoisingAutoEncoder, AutoEncoder, Dense, Activation, TimeDistributedDense, Flatten
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.layers.core import Layer
from keras.layers import containers
from keras.utils import np_utils

import numpy as np

# Try different things here: 'lstm' or 'classical' or 'denoising'
autoencoder_type = 'denoising'

nb_classes = 10
batch_size = 128
nb_epoch = 5
activation = 'linear'

input_dim = 784
hidden_dim = 392

max_train_samples = 5000
max_test_samples = 1000

# the data, shuffled and split between tran and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000, input_dim)[:max_train_samples]
X_test = X_test.reshape(10000, input_dim)[:max_test_samples]
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
X_train /= 255
X_test /= 255

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)[:max_train_samples]
Y_test = np_utils.to_categorical(y_test, nb_classes)[:max_test_samples]

print("X_train: ", X_train.shape)
print("X_test: ", X_test.shape)

##########################
# dense model test       #
##########################

print("Training classical fully connected layer for classification")
model_classical = Sequential()
model_classical.add(Dense(input_dim, 10, activation=activation))
model_classical.add(Activation('softmax'))
model_classical.get_config(verbose=1)
model_classical.compile(loss='categorical_crossentropy', optimizer='adam')
model_classical.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=False, verbose=0, validation_data=(X_test, Y_test))
classical_score = model_classical.evaluate(X_test, Y_test, verbose=0, show_accuracy=True)
print('\nclassical_score:', classical_score)

##########################
# autoencoder model test #
##########################

def build_lstm_autoencoder(autoencoder, X_train, X_test):
	X_train = X_train[:, np.newaxis, :] 
	X_test = X_test[:, np.newaxis, :]
	print("Modified X_train: ", X_train.shape)
	print("Modified X_test: ", X_test.shape)

	# The TimeDistributedDense isn't really necessary, however you need a lot of GPU memory to do 784x394-394x784
	autoencoder.add(TimeDistributedDense(input_dim, 16))
	autoencoder.add(AutoEncoder(encoder=LSTM(16, 8, activation=activation, return_sequences=True),
								decoder=LSTM(8, input_dim, activation=activation, return_sequences=True),
								output_reconstruction=False, tie_weights=True))
	return autoencoder, X_train, X_test

def build_deep_classical_autoencoder(autoencoder):
	encoder = containers.Sequential([Dense(input_dim, hidden_dim, activation=activation), Dense(hidden_dim, hidden_dim/2, activation=activation)])
	decoder = containers.Sequential([Dense(hidden_dim/2, hidden_dim, activation=activation), Dense(hidden_dim, input_dim, activation=activation)])
	autoencoder.add(AutoEncoder(encoder=encoder, decoder=decoder, output_reconstruction=False, tie_weights=True))
	return autoencoder

def build_denoising_autoencoder(autoencoder):
	# You need another layer before a denoising autoencoder
	# This is similar to the dropout layers, etc..
	autoencoder.add(Dense(input_dim, input_dim))
	autoencoder.add(DenoisingAutoEncoder(encoder=Dense(input_dim, hidden_dim, activation=activation),
										 decoder=Dense(hidden_dim, input_dim, activation=activation),
										 output_reconstruction=False, tie_weights=True, corruption_level=0.3))
	return autoencoder

# Build our autoencoder model
autoencoder = Sequential()
if autoencoder_type == 'lstm':
	print("Training LSTM AutoEncoder")
	autoencoder, X_train, X_test = build_lstm_autoencoder(autoencoder, X_train, X_test)
elif autoencoder_type == 'denoising':
	print("Training Denoising AutoEncoder")
	autoencoder = build_denoising_autoencoder(autoencoder)
elif autoencoder_type == 'classical':
	print("Training Classical AutoEncoder")
	autoencoder = build_deep_classical_autoencoder(autoencoder)
else:
	print("Error: unknown autoencoder type!")
	exit(-1)

autoencoder.get_config(verbose=1)
autoencoder.compile(loss='mean_squared_error', optimizer='adam')
# Do NOT use validation data with return output_reconstruction=True
autoencoder.fit(X_train, X_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=False, verbose=1)

# Do an inference pass
prefilter_train = autoencoder.predict(X_train, verbose=0)
prefilter_test = autoencoder.predict(X_test, verbose=0)
print("prefilter_train: ", prefilter_train.shape)
print("prefilter_test: ", prefilter_test.shape)

# Classify results from Autoencoder
print("Building classical fully connected layer for classification")
model = Sequential()
if autoencoder_type == 'lstm':
	model.add(TimeDistributedDense(8, nb_classes, activation=activation))
	model.add(Flatten())
elif autoencoder_type == 'classical':
	model.add(Dense(prefilter_train.shape[1], nb_classes, activation=activation))
else:
	model.add(Dense(prefilter_train.shape[1], nb_classes, activation=activation))

model.add(Activation('softmax'))

model.get_config(verbose=1)
model.compile(loss='categorical_crossentropy', optimizer='adam')
model.fit(prefilter_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=False, verbose=0, validation_data=(prefilter_test, Y_test))

score = model.evaluate(prefilter_test, Y_test, verbose=0, show_accuracy=True)
print('\nscore:', score)

print('Loss change:', (score[0] - classical_score[0])/classical_score[0], '%')
print('Accuracy change:', (score[1] - classical_score[1])/classical_score[1], '%')

