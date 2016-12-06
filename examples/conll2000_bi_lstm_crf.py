'''This example demonstrates the use of a bidirectional LSTM and a
Linear Chain Conditional Random Field for text chunking.
(http://www.cnts.ua.ac.be/conll2000/chunking/).

Gets >= 93 FB1 score on test dataset after 3 epochs.

'''
from __future__ import print_function, unicode_literals
import numpy as np
np.random.seed(1337)  # for reproducibility

from six.moves import zip
from keras.preprocessing import sequence
from keras.models import Model
from keras.layers import Input, merge
from keras.layers import Dense, Embedding, ChainCRF, LSTM, Bidirectional, Dropout
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import RMSprop
from keras.datasets import conll2000
from keras.utils.data_utils import get_file
from keras.callbacks import Callback
from subprocess import Popen, PIPE, STDOUT


def run_conlleval(X_words_test, y_test, y_pred, index2word, index2chunk, pad_id=0):
    '''
    Runs the conlleval script for evaluation the predicted IOB-tags.
    '''
    url = 'http://www.cnts.ua.ac.be/conll2000/chunking/conlleval.txt'
    path = get_file('conlleval',
                    origin=url,
                    md5_hash='61b632189e5a05d5bd26a2e1ec0f4f9e')

    p = Popen(['perl', path], stdout=PIPE, stdin=PIPE, stderr=STDOUT)

    y_true = np.squeeze(y_test, axis=2)

    sequence_lengths = np.argmax(X_words_test == pad_id, axis=1)
    nb_samples = X_words_test.shape[0]
    conlleval_input = []
    for k in range(nb_samples):
        sent_len = sequence_lengths[k]
        words = list(map(lambda idx: index2word[idx], X_words_test[k][:sent_len]))
        true_tags = list(map(lambda idx: index2chunk[idx], y_true[k][:sent_len]))
        pred_tags = list(map(lambda idx: index2chunk[idx], y_pred[k][:sent_len]))
        sent = zip(words, true_tags, pred_tags)
        for row in sent:
            conlleval_input.append(' '.join(row))
        conlleval_input.append('')
    print()
    conlleval_stdout = p.communicate(input='\n'.join(conlleval_input).encode())[0]
    print(conlleval_stdout.decode())


class ConllevalCallback(Callback):
    '''Callback for running the conlleval script on the test dataset after
    each epoch.
    '''
    def __init__(self, X_test, y_test, batch_size=1, index2word=None, index2chunk=None):
        self.X_words_test, self.X_pos_test = X_test
        self.y_test = y_test
        self.batch_size = batch_size
        self.index2word = index2word
        self.index2chunk = index2chunk

    def on_epoch_end(self, epoch, logs={}):
        X_test = [self.X_words_test, self.X_pos_test]
        pred_proba = model.predict(X_test)
        y_pred = np.argmax(pred_proba, axis=2)
        run_conlleval(self.X_words_test, self.y_test, y_pred, self.index2word, self.index2chunk)


maxlen = 80  # cut texts after this number of words (among top max_features most common words)
word_embedding_dim = 100
pos_embedding_dim = 32
lstm_dim = 100
batch_size = 64

print('Loading data...')
(X_words_train, X_pos_train, y_train), (X_words_test, X_pos_test, y_test), (index2word, index2pos, index2chunk) = conll2000.load_data(word_preprocess=lambda w: w.lower())

max_features = len(index2word)
nb_pos_tags = len(index2pos)
nb_chunk_tags = len(index2chunk)

X_words_train = sequence.pad_sequences(X_words_train, maxlen=maxlen, padding='post')
X_pos_train = sequence.pad_sequences(X_pos_train, maxlen=maxlen, padding='post')
X_words_test = sequence.pad_sequences(X_words_test, maxlen=maxlen, padding='post')
X_pos_test = sequence.pad_sequences(X_pos_test, maxlen=maxlen, padding='post')
y_train = sequence.pad_sequences(y_train, maxlen=maxlen, padding='post')
y_train = np.expand_dims(y_train, -1)
y_test = sequence.pad_sequences(y_test, maxlen=maxlen, padding='post')
y_test = np.expand_dims(y_test, -1)

print('Unique words:', max_features)
print('Unique pos_tags:', nb_pos_tags)
print('Unique chunk tags:', nb_chunk_tags)
print('X_words_train shape:', X_words_train.shape)
print('X_words_test shape:', X_words_test.shape)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)

print('Build model...')

word_input = Input(shape=(maxlen,), dtype='int32', name='word_input')
word_emb = Embedding(max_features, word_embedding_dim, input_length=maxlen, dropout=0.2, name='word_emb')(word_input)
pos_input = Input(shape=(maxlen,), dtype='int32', name='pos_input')
pos_emb = Embedding(nb_pos_tags, pos_embedding_dim, input_length=maxlen, dropout=0.2, name='pos_emb')(pos_input)
total_emb = merge([word_emb, pos_emb], mode='concat', concat_axis=2)

bilstm = Bidirectional(LSTM(lstm_dim, dropout_W=0.2, dropout_U=0.2, return_sequences=True))(total_emb)
bilstm_d = Dropout(0.2)(bilstm)
dense = TimeDistributed(Dense(nb_chunk_tags))(bilstm_d)

crf = ChainCRF()
crf_output = crf(dense)

model = Model(input=[word_input, pos_input], output=[crf_output])

model.compile(loss=crf.sparse_loss,
              optimizer=RMSprop(0.01),
              metrics=['sparse_categorical_accuracy'])

model.summary()


conlleval = ConllevalCallback([X_words_test, X_pos_test], y_test,
                              index2word=index2word, index2chunk=index2chunk,
                              batch_size=batch_size)
print('Train...')
model.fit([X_words_train, X_pos_train], y_train,
          batch_size=batch_size, nb_epoch=3, callbacks=[conlleval])
