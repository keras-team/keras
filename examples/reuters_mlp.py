from __future__ import absolute_import
from __future__ import print_function
import numpy as np

from keras.datasets import reuters
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.preprocessing.text import Tokenizer

'''
    Train and evaluate a simple MLP on the Reuters newswire topic classification task.

    GPU run command:
        THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python examples/reuters_mlp.py

    CPU run command:
        python examples/reuters_mlp.py
'''

max_words = 10000
batch_size = 16

print("Loading data...")
(X_train, y_train), (X_test, y_test) = reuters.load_data(nb_words=max_words, test_split=0.2)
print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')

nb_classes = np.max(y_train)+1
print(nb_classes, 'classes')

print("Vectorizing sequence data...")
tokenizer = Tokenizer(nb_words=max_words)
X_train = tokenizer.sequences_to_matrix(X_train, mode="binary")
X_test = tokenizer.sequences_to_matrix(X_test, mode="binary")
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

print("Convert class vector to binary class matrix (for use with categorical_crossentropy)")
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)
print('Y_train shape:', Y_train.shape)
print('Y_test shape:', Y_test.shape)

print("Building model...")
model = Sequential()
model.add(Dense(max_words, 256, init='normal'))
model.add(Activation('relu'))
model.add(BatchNormalization(input_shape=(256,))) # try without batch normalization (doesn't work as well!)
model.add(Dropout(0.5))
model.add(Dense(256, nb_classes, init='normal'))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')

# import cPickle
# model = cPickle.load(open('testsave.m.pkl'))

for v in range(3):
    for sa in [True, False]:
        for vs in [0, 0.1]:
            print('='*40)
            print('v:%d, sa:%r, vs:%f' % (v, sa, vs))
            print("Training...")
            model.fit(X_train, Y_train, nb_epoch=2, batch_size=batch_size, verbose=v, show_accuracy=sa, validation_split=vs)
            score = model.evaluate(X_test, Y_test, batch_size=batch_size, verbose=v, show_accuracy=sa)
            print('Test score:', score)

            classes = model.predict_classes(X_test, batch_size=batch_size, verbose=v)
            acc = np_utils.accuracy(classes, y_test)
            print('Test accuracy:', acc)

# model.save('testsave.m')


