'''Train a Bidirectional Two-level GRU on the IMDB sentiment classification task.

Output after 1 epoch on CPU: ~0.8535
Time per epoch on CPU (Core i7): ~600s.

Exercise ideas:
    * Evaluate on a larger dataset (or with larger training fraction on imdb)
    * That Lambda() in aggregation functions sure look scary, would you
      get the same results if you set return_sequences=False in the last
      GRU level?
    * Try to disable the skip-layer connections
    * GRU is a popular alternative to LSTM as it is conceptually simpler,
      easier to train and has the same performance on many NLP tasks; what
      about on this task?
    * Should the RNN output dimensionality be smaller, same or bigger than
      input dimensionality?
    * We combine RNN direction outputs by summing them (to get the same
      dimensionality everywhere); a more usual approach in the literature
      is to concatenate them.  Does it matter?
    * Preinitialize all the GRUs to by default just copy the embeddings
      (init='identity'), i.e. start with essentially a bag-of-words model
    * Try different aggregation functions
    * The current model clearly overfits with more epochs; would it be
      better to apply (perhaps smaller) dropout at each level?
    * How much does the final accuracy vary when you disable the fixed
      random seed and simply re-run the script 4 or 8 times?  Compute
      confidence intervals using Student's t-distribution.  What are the
      implications to your findings above?
'''

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.models import Model
from keras.layers import Dense, Dropout, Embedding, GRU, Input, Lambda, merge
from keras.datasets import imdb


max_features = 20000
maxlen = 100  # cut texts after this number of words (among top max_features most common words)
batch_size = 32
N = 128  # base dimensionality of layers
rnn_levels = 2


def reverse_time(x):
    import keras.backend as K
    assert K.ndim(x) == 3, "Should be a 3D tensor."
    rev = K.permute_dimensions(x, (1, 0, 2))[::-1]
    return K.permute_dimensions(rev, (1, 0, 2))

def time_first(x):
    import keras.backend as K
    assert K.ndim(x) == 3, "Should be a 3D tensor."
    return K.permute_dimensions(x, (1, 0, 2))[0]

def time_last(x):
    import keras.backend as K
    assert K.ndim(x) == 3, "Should be a 3D tensor."
    return K.permute_dimensions(x, (1, 0, 2))[-1]

def time_one_shape(inp_shape):
    return (inp_shape[0], inp_shape[2])


print('Loading data...')
(X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=max_features)
print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')

print("Pad sequences (samples x time)")
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)
y_train = np.array(y_train)
y_test = np.array(y_test)


# this is the placeholder tensor for the input sequences
sequence = Input(shape=(maxlen,), dtype='int32')
# this embedding layer will transform the sequences of integers
# into vectors of size 128
embedded_seq = Embedding(max_features, N, input_length=maxlen)(sequence)


# Multi-level RNN:
for i in range(rnn_levels):
    if i == 0:
        rnn_input = embedded_seq
    else:
        # compose previous forwards and backwards outputs for each token
        processed_seq = merge([forwards, backwards], mode='sum')

        # Skip-layer connections is a popular enhancement for multi-level
        # RNN that feed both the previous layer and the original input
        # to the next level
        rnn_input = merge([processed_seq, embedded_seq], mode='sum')

    # apply forwards GRU
    forwards = GRU(N, return_sequences=True)(rnn_input)
    # apply backwards GRU
    backwards = GRU(N, return_sequences=True, go_backwards=True)(rnn_input)

    # the backwards sequence is in reverse order, so we need to flip it
    # if we are to element-wise merge it with forwards!
    backwards = Lambda(reverse_time)(backwards)


# Aggregation Function:
# We need to get a "summary" fixed-size embedding from the sequences
# produced by the RNN.  There are several options:
# * Get the final embedding produced by the RNN (last token for
#   forwards, first token for backwards).  This is shown here and
#   the same as if we set return_sequences=False for the last GRU level.
# * MaxPooling1D with pool_length=maxlen (i.e. element-wise max over
#   the sequence)
# * AveragePooling1D or other variations.
forwards1 = Lambda(time_last, output_shape=time_one_shape)(forwards)
backwards1 = Lambda(time_first, output_shape=time_one_shape)(backwards)
aggregate = merge([forwards1, backwards1], mode='sum')


after_dp = Dropout(0.5)(aggregate)
output = Dense(1, activation='sigmoid')(after_dp)

model = Model(input=sequence, output=output)

# try using different optimizers and different optimizer configs
model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

print('Train...')
model.fit(X_train, y_train,
          batch_size=batch_size,
          nb_epoch=4,
          validation_data=[X_test, y_test])
