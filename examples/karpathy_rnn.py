# Implementation of a simple character RNN (using LSTM units), based on:
# https://github.com/karpathy/char-rnn
import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, TimeDistributedDense
from keras.layers.recurrent import LSTM

text = open('input.txt', 'r').read()
char_to_idx = {ch: i for (i, ch) in enumerate(sorted(list(set(text))))}
idx_to_char = {i: ch for (ch, i) in char_to_idx.items()}
vocab_size = len(char_to_idx)

print 'Working on %d characters (%d unique)' % (len(text), vocab_size)

SEQ_LEN = 64
BATCH_SIZE = 16
BATCH_CHARS = len(text) / BATCH_SIZE
LSTM_SIZE = 128
LAYERS = 3


# For training, each subsequent example for a given batch index should be a
# consecutive portion of the text.  To achieve this, each batch index operates
# over a disjoint section of the input text.
def read_batches(text):
    T = np.asarray([char_to_idx[c] for c in text], dtype=np.int32)
    X = np.zeros((BATCH_SIZE, SEQ_LEN, vocab_size))
    Y = np.zeros((BATCH_SIZE, SEQ_LEN, vocab_size))

    for i in range(0, BATCH_CHARS - SEQ_LEN - 1, SEQ_LEN):
        X[:] = 0
        Y[:] = 0
        for batch_idx in range(BATCH_SIZE):
            start = batch_idx * BATCH_CHARS + i
            for j in range(SEQ_LEN):
                X[batch_idx, j, T[start+j]] = 1
                Y[batch_idx, j, T[start+j+1]] = 1

        yield X, Y


def build_model(infer):
    if infer:
        batch_size = seq_len = 1
    else:
        batch_size = BATCH_SIZE
        seq_len = SEQ_LEN
    model = Sequential()
    model.add(LSTM(LSTM_SIZE,
                   return_sequences=True,
                   batch_input_shape=(batch_size, seq_len, vocab_size),
                   stateful=True))

    model.add(Dropout(0.2))
    for l in range(LAYERS - 1):
        model.add(LSTM(LSTM_SIZE, return_sequences=True, stateful=True))
        model.add(Dropout(0.2))

    model.add(TimeDistributedDense(vocab_size))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adagrad')
    return model


print 'Building model.'
training_model = build_model(infer=False)
test_model = build_model(infer=True)
print '... done'


def sample(epoch, sample_chars=256, primer_text='And the '):
    test_model.reset_states()
    test_model.load_weights('/tmp/keras_char_rnn.%d.h5' % epoch)
    sampled = [char_to_idx[c] for c in primer_text]

    for c in primer_text:
        batch = np.zeros((1, 1, vocab_size))
        batch[0, 0, char_to_idx[c]] = 1
        test_model.predict_on_batch(batch)

    for i in range(sample_chars):
        batch = np.zeros((1, 1, vocab_size))
        batch[0, 0, sampled[-1]] = 1
        softmax = test_model.predict_on_batch(batch)[0].ravel()
        sample = np.random.choice(range(vocab_size), p=softmax)
        sampled.append(sample)

    print ''.join([idx_to_char[c] for c in sampled])

for epoch in range(100):
    for i, (x, y) in enumerate(read_batches(text)):
        loss = training_model.train_on_batch(x, y)
        print epoch, i, loss

        if i % 1000 == 0:
            training_model.save_weights('/tmp/keras_char_rnn.%d.h5' % epoch,
                                        overwrite=True)
            sample(epoch)
