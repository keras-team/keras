'''The example demonstrates how to write custom layers with mask supporting for Keras.

We build a custom layer called 'AverageEmbeddingLayer',
which takes the mask into account.

We need to specify three methods: `compute_output_shape`, `call` and `compute_mask`.

The `call` and `compute_mask` accept a second argument `mask`, which
is the mask of the previous layer

Note that the same result can also be achieved via a Lambda layer.
'''
from __future__ import print_function
import keras
from keras import backend as K
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Layer
from keras.datasets import imdb


class AverageEmbeddingLayer(Layer):
    '''This layer average the word vectors of input sequence
    over time. The 0 input will be mask out, and not be considered

    For example, let the sequence be [1, 3, 2, 0, 0],
    the output of embedding will be (w1, w3, w2, w0, w0)
    the output of this layer will be (w1 + w3 + w2) / 3

    # Input shape
        3D tensor of shape (samples, time, feature)

    # Input mask
        2D tensor of shape (samples, time)

    # Output shape
        2D tensor of shape (samples, feature)

    # Output mask
        None
    '''
    def __init__(self, **kwargs):
        super(AverageEmbeddingLayer, self).__init__(**kwargs)
        self.supports_masking = True
    
    def call(self, x, mask=None):
        assert mask is not None
        mask = K.cast(mask, K.dtype(x))
        
        length = K.expand_dims(K.sum(mask, axis=1), axis=-1)  # (samples, 1)
        
        return K.sum(x * K.expand_dims(mask, axis=-1), axis=1) / length

    def compute_output_shape(self, input_shape):
        assert len(input_shape) == 3

        return (input_shape[0], input_shape[2])
    
    def compute_mask(self, inputs, mask=None):
        return None


# set parameters
max_features = 5000
maxlen = 2500
batch_size = 32
embedding_dims = 100
hidden_dims = 250
epochs = 2

# the data
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

print(x_train.shape)
print(x_test.shape)

# build the model (based almostly on imdb_cnn.py)
model = Sequential()
model.add(Embedding(max_features,
                    embedding_dims,
                    input_length=maxlen,
                    mask_zero=True))
model.add(AverageEmbeddingLayer())
model.add(Dropout(0.2))

# We add a vanilla hidden layer:
model.add(Dense(hidden_dims))
model.add(Dropout(0.2))
model.add(Activation('relu'))

# We project onto a single unit output layer, and squash it with a sigmoid:
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test))
