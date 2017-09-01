'''This example demonstrates the use of Convolution1D for text classification using the functional API.

Gets to 0.89 test accuracy after 2 epochs.
90s/epoch on Intel i5 2.4Ghz CPU.
10s/epoch on Tesla K40 GPU.

'''

from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Model
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding, Input
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.datasets import imdb

# set parameters:
max_features = 5000
maxlen = 400
batch_size = 32
embedding_dims = 50
filters = 250
kernel_size = 3
hidden_dims = 250
epochs = 2

print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print('Build model...')

# define the input data
input_data = Input(shape=(maxlen,))

# we add an efficient embedding layer which maps
# our vocab indices into embedding_dims dimensions
x = Embedding(max_features,
              embedding_dims,
              input_length=maxlen)(input_data)
x = Dropout(0.2)(x)

# we add a Convolution1D, which will learn filters
# word group filters of size filter_length:
x = Conv1D(filters,
           kernel_size,
           padding='valid',
           activation='relu',
           strides=1)(x)
# we use max pooling:
x = GlobalMaxPooling1D()(x)

# We add a vanilla hidden layer:
x = Dense(hidden_dims)(x)
x = Dropout(0.2)(x)
x = Activation('relu')(x)

# We project onto a single unit output layer, and squash it with a sigmoid:
x = Dense(1)(x)
out = Activation('sigmoid')(x)


model = Model(input_data, out)

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
