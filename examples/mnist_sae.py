from __future__ import absolute_import
from __future__ import print_function
import numpy as np
np.random.seed(1337) # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import containers
from keras.layers.core import Dense, AutoEncoder
from keras.utils import np_utils

'''
    Example of Stacked Autoencoder.
    This is not an interesting example but to demonstrate how to use AutoEncoder
    module and do layer-wise pre-training and fine-tuning
'''

batch_size = 64
nb_classes = 10
nb_epoch = 3
nb_hidden_layers = [784, 600, 500, 400]

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(-1, 784)
X_test = X_test.reshape(-1, 784)
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

# Layer-wise pre-training
trained_encoders = []
X_train_tmp = X_train
for n_in, n_out in zip(nb_hidden_layers[:-1], nb_hidden_layers[1:]):
    print('Pre-training the layer: Input {} -> Output {}'.format(n_in, n_out))
    # Create AE and training
    ae = Sequential()
    encoder = containers.Sequential([Dense(n_in, n_out, activation='sigmoid')])
    decoder = containers.Sequential([Dense(n_out, n_in, activation='sigmoid')])
    ae.add(AutoEncoder(encoder=encoder, decoder=decoder,
                       output_reconstruction=False, tie_weights=True))
    ae.compile(loss='mean_squared_error', optimizer='rmsprop')
    ae.fit(X_train_tmp, X_train_tmp, batch_size=batch_size, nb_epoch=nb_epoch)
    # Store trainined weight
    trained_encoders.append(ae.layers[0].encoder)
    # Update training data
    X_train_tmp = ae.predict(X_train_tmp)

# Fine-tuning
print('Fine-tuning')
model = Sequential()
for encoder in trained_encoders:
    model.add(encoder)
model.add(Dense(nb_hidden_layers[-1], nb_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          show_accuracy=True, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
