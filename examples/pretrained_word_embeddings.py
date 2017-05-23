'''This script loads pre-trained word embeddings (GloVe embeddings)
into a frozen Keras Embedding layer, and uses it to
train a text classification model on the 20 Newsgroup dataset
(classication of newsgroup messages into 20 different categories).

GloVe embedding data can be found at:
http://nlp.stanford.edu/data/glove.6B.zip
(source page: http://nlp.stanford.edu/projects/glove/)

20 Newsgroup data can be found at:
http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-20/www/data/news20.html
'''

from __future__ import print_function

import os
import sys
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model


BASE_DIR = ''
GLOVE_DIR = BASE_DIR + '/glove.6B/'
TEXT_DATA_DIR = BASE_DIR + '/20_newsgroup/'
MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2

# first, build index mapping words in the embeddings set
# to their embedding vector

print('Indexing word vectors.')

#load embedding file, create dict
embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

# second, prepare text samples and their labels
print('Processing text dataset')

texts = []  # list of text samples
labels_index = {}  # dictionary mapping label name to numeric id
labels = []  # list of label ids
for name in sorted(os.listdir(TEXT_DATA_DIR)):
    path = os.path.join(TEXT_DATA_DIR, name)
    if os.path.isdir(path):
        label_id = len(labels_index)
        labels_index[name] = label_id
        for fname in sorted(os.listdir(path)):
            if fname.isdigit():
                fpath = os.path.join(path, fname)
                if sys.version_info < (3,):
                    f = open(fpath)
                else:
                    f = open(fpath, encoding='latin-1')
                t = f.read()
                i = t.find('\n\n')  # skip header
                if 0 < i:
                    t = t[i:]
                texts.append(t)
                f.close()
                labels.append(label_id)

print('Found %s texts.' % len(texts))

# finally, vectorize the text samples into a 2D integer tensor
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
# fit texts, assign an id to each word
tokenizer.fit_on_texts(texts)
# convert sentence to a word id sequence
# sequences is a id sequence
sequences = tokenizer.texts_to_sequences(texts)
# word_index:word->id dict,eg: {'a': 1, 'boy': 2, 'c': 3, 'b': 4, 'cat': 5, 'bad': 6, 'good': 7}
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

# pad_sequences: 将长为nb_samples的序列（标量序列）转化为形如(nb_samples,nb_timesteps)2D numpy array
# 如果提供了参数maxlen，nb_timesteps=maxlen，否则其值为最长序列的长度。其他短于该长度的序列都会在后部填充0以达到该长度。长于nb_timesteps的序列将会被截断，以使其匹配目标长度
# keras.preprocessing.sequence.pad_sequences(sequences, maxlen=None, dtype='int32', padding='pre', truncating='pre', value=0.)
# if sequences = [[1, 7, 2], [1, 6, 5]] after pad_sequences data = pad_sequences(sequences, maxlen=5), result is :array([[0, 0, 1, 7, 2],[0, 0, 1, 6, 5]], dtype=int32),left fullfil with zero '0'
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

# convert category from value to one-hot vector
# 	>>> labels[1]=1
# 	>>> labels[2]=2
# 	>>> labels[0]=0
# 	>>> labels
# 	[0, 1, 2]
# 	>>> to_categorical(np.asarray(labels))
# 	array([[ 1.,  0.,  0.],
#       	[ 0.,  1.,  0.],
#       	[ 0.,  0.,  1.]])
labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

# split the data into a training set and a validation set
# data.shape[0] is row index
indices = np.arange(data.shape[0])
# shuffle row index
np.random.shuffle(indices)
# shuffle data according to row index
data = data[indices]
# shuffle labels according to row index
labels = labels[indices]

num_validation_samples = int(VALIDATION_SPLIT * data.shape[0])
# x_train=0:num_validation_samples
x_train = data[:-num_validation_samples]
y_train = labels[:-num_validation_samples]
# x_val=num_validation_samples:end
x_val = data[-num_validation_samples:]
y_val = labels[-num_validation_samples:]

print('Preparing embedding matrix.')

# prepare embedding matrix
num_words = min(MAX_NB_WORDS, len(word_index))
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if i >= MAX_NB_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

# load pre-trained word embeddings into an Embedding layer
# note that we set trainable = False so as to keep the embeddings fixed
embedding_layer = Embedding(num_words,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)

print('Training model.')

# train a 1D convnet with global maxpooling
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
x = Conv1D(128, 5, activation='relu')(embedded_sequences)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(35)(x)
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
preds = Dense(len(labels_index), activation='softmax')(x)

model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

model.fit(x_train, y_train,
          batch_size=128,
          epochs=10,
          validation_data=(x_val, y_val))
