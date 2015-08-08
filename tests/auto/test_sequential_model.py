from __future__ import absolute_import
from __future__ import print_function
import unittest
import numpy as np
np.random.seed(1337)

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Merge
from keras.utils import np_utils
from keras.utils.test_utils import get_test_data

input_dim = 32
nb_hidden = 16
nb_class = 4
batch_size = 64
nb_epoch = 1

train_samples = 5000
test_samples = 1000

(X_train, y_train), (X_test, y_test) = get_test_data(nb_train=train_samples, nb_test=test_samples, input_shape=(input_dim,),
                                                     classification=True, nb_class=4)
y_test = np_utils.to_categorical(y_test)
y_train = np_utils.to_categorical(y_train)
print(X_train.shape)
print(y_train.shape)


class TestSequential(unittest.TestCase):
    def test_sequential(self):
        print('Test sequential')
        model = Sequential()
        model.add(Dense(input_dim, nb_hidden))
        model.add(Activation('relu'))
        model.add(Dense(nb_hidden, nb_class))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

        model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=1, validation_data=(X_test, y_test))
        model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=False, verbose=2, validation_data=(X_test, y_test))
        model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=2, validation_split=0.1)
        model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=False, verbose=1, validation_split=0.1)
        model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=0)
        model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, shuffle=False)

        model.train_on_batch(X_train[:32], y_train[:32])

        loss = model.evaluate(X_train, y_train, verbose=0)
        print('loss:', loss)
        if loss > 0.6:
            raise Exception('Score too low, learning issue.')
        preds = model.predict(X_test, verbose=0)
        classes = model.predict_classes(X_test, verbose=0)
        probas = model.predict_proba(X_test, verbose=0)
        print(model.get_config(verbose=1))

        print('test weight saving')
        model.save_weights('temp.h5', overwrite=True)
        model = Sequential()
        model.add(Dense(input_dim, nb_hidden))
        model.add(Activation('relu'))
        model.add(Dense(nb_hidden, nb_class))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
        model.load_weights('temp.h5')

        nloss = model.evaluate(X_train, y_train, verbose=0)
        print(nloss)
        assert(loss == nloss)

    def test_merge_sum(self):
        print('Test merge: sum')
        left = Sequential()
        left.add(Dense(input_dim, nb_hidden))
        left.add(Activation('relu'))

        right = Sequential()
        right.add(Dense(input_dim, nb_hidden))
        right.add(Activation('relu'))

        model = Sequential()
        model.add(Merge([left, right], mode='sum'))

        model.add(Dense(nb_hidden, nb_class))
        model.add(Activation('softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

        model.fit([X_train, X_train], y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=0, validation_data=([X_test, X_test], y_test))
        model.fit([X_train, X_train], y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=False, verbose=0, validation_data=([X_test, X_test], y_test))
        model.fit([X_train, X_train], y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=0, validation_split=0.1)
        model.fit([X_train, X_train], y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=False, verbose=0, validation_split=0.1)
        model.fit([X_train, X_train], y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=0)
        model.fit([X_train, X_train], y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=0, shuffle=False)

        loss = model.evaluate([X_train, X_train], y_train, verbose=0)
        print('loss:', loss)
        if loss > 0.7:
            raise Exception('Score too low, learning issue.')
        preds = model.predict([X_test, X_test], verbose=0)
        classes = model.predict_classes([X_test, X_test], verbose=0)
        probas = model.predict_proba([X_test, X_test], verbose=0)
        print(model.get_config(verbose=1))

        print('test weight saving')
        model.save_weights('temp.h5', overwrite=True)
        left = Sequential()
        left.add(Dense(input_dim, nb_hidden))
        left.add(Activation('relu'))
        right = Sequential()
        right.add(Dense(input_dim, nb_hidden))
        right.add(Activation('relu'))
        model = Sequential()
        model.add(Merge([left, right], mode='sum'))
        model.add(Dense(nb_hidden, nb_class))
        model.add(Activation('softmax'))
        model.load_weights('temp.h5')
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

        nloss = model.evaluate([X_train, X_train], y_train, verbose=0)
        print(nloss)
        assert(loss == nloss)

    def test_merge_concat(self):
        print('Test merge: concat')
        left = Sequential()
        left.add(Dense(input_dim, nb_hidden))
        left.add(Activation('relu'))

        right = Sequential()
        right.add(Dense(input_dim, nb_hidden))
        right.add(Activation('relu'))

        model = Sequential()
        model.add(Merge([left, right], mode='concat'))

        model.add(Dense(nb_hidden * 2, nb_class))
        model.add(Activation('softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

        model.fit([X_train, X_train], y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=0, validation_data=([X_test, X_test], y_test))
        model.fit([X_train, X_train], y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=False, verbose=0, validation_data=([X_test, X_test], y_test))
        model.fit([X_train, X_train], y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=0, validation_split=0.1)
        model.fit([X_train, X_train], y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=False, verbose=0, validation_split=0.1)
        model.fit([X_train, X_train], y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=0)
        model.fit([X_train, X_train], y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=0, shuffle=False)

        loss = model.evaluate([X_train, X_train], y_train, verbose=0)
        print('loss:', loss)
        if loss > 0.6:
            raise Exception('Score too low, learning issue.')
        preds = model.predict([X_test, X_test], verbose=0)
        classes = model.predict_classes([X_test, X_test], verbose=0)
        probas = model.predict_proba([X_test, X_test], verbose=0)
        print(model.get_config(verbose=1))

        print('test weight saving')
        model.save_weights('temp.h5', overwrite=True)
        left = Sequential()
        left.add(Dense(input_dim, nb_hidden))
        left.add(Activation('relu'))

        right = Sequential()
        right.add(Dense(input_dim, nb_hidden))
        right.add(Activation('relu'))

        model = Sequential()
        model.add(Merge([left, right], mode='concat'))

        model.add(Dense(nb_hidden * 2, nb_class))
        model.add(Activation('softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
        model.load_weights('temp.h5')

        nloss = model.evaluate([X_train, X_train], y_train, verbose=0)
        print(nloss)
        assert(loss == nloss)

    def test_merge_recursivity(self):
        print('Test merge recursivity')

        left = Sequential()
        left.add(Dense(input_dim, nb_hidden))
        left.add(Activation('relu'))

        right = Sequential()
        right.add(Dense(input_dim, nb_hidden))
        right.add(Activation('relu'))

        righter = Sequential()
        righter.add(Dense(input_dim, nb_hidden))
        righter.add(Activation('relu'))

        intermediate = Sequential()
        intermediate.add(Merge([left, right], mode='sum'))
        intermediate.add(Dense(nb_hidden, nb_hidden))
        intermediate.add(Activation('relu'))

        model = Sequential()
        model.add(Merge([intermediate, righter], mode='sum'))

        model.add(Dense(nb_hidden, nb_class))
        model.add(Activation('softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

        model.fit([X_train, X_train, X_train], y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=0, validation_data=([X_test, X_test, X_test], y_test))
        model.fit([X_train, X_train, X_train], y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=False, verbose=0, validation_data=([X_test, X_test, X_test], y_test))
        model.fit([X_train, X_train, X_train], y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=0, validation_split=0.1)
        model.fit([X_train, X_train, X_train], y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=False, verbose=0, validation_split=0.1)
        model.fit([X_train, X_train, X_train], y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=0)
        model.fit([X_train, X_train, X_train], y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=0, shuffle=False)

        loss = model.evaluate([X_train, X_train, X_train], y_train, verbose=0)
        print('loss:', loss)
        if loss > 0.6:
            raise Exception('Score too low, learning issue.')
        preds = model.predict([X_test, X_test, X_test], verbose=0)
        classes = model.predict_classes([X_test, X_test, X_test], verbose=0)
        probas = model.predict_proba([X_test, X_test, X_test], verbose=0)
        print(model.get_config(verbose=1))

        model.save_weights('temp.h5', overwrite=True)
        model.load_weights('temp.h5')

        nloss = model.evaluate([X_train, X_train, X_train], y_train, verbose=0)
        print(nloss)
        assert(loss == nloss)

    def test_merge_overlap(self):
        print('Test merge overlap')
        left = Sequential()
        left.add(Dense(input_dim, nb_hidden))
        left.add(Activation('relu'))

        model = Sequential()
        model.add(Merge([left, left], mode='sum'))

        model.add(Dense(nb_hidden, nb_class))
        model.add(Activation('softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

        model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=1, validation_data=(X_test, y_test))
        model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=False, verbose=2, validation_data=(X_test, y_test))
        model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=2, validation_split=0.1)
        model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=False, verbose=1, validation_split=0.1)
        model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=0)
        model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, shuffle=False)

        model.train_on_batch(X_train[:32], y_train[:32])

        loss = model.evaluate(X_train, y_train, verbose=0)
        print('loss:', loss)
        if loss > 0.6:
            raise Exception('Score too low, learning issue.')
        preds = model.predict(X_test, verbose=0)
        classes = model.predict_classes(X_test, verbose=0)
        probas = model.predict_proba(X_test, verbose=0)
        print(model.get_config(verbose=1))

        model.save_weights('temp.h5', overwrite=True)
        model.load_weights('temp.h5')

        nloss = model.evaluate(X_train, y_train, verbose=0)
        print(nloss)
        assert(loss == nloss)


if __name__ == '__main__':
    print('Test Sequential model')
    unittest.main()
