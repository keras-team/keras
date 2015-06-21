from __future__ import absolute_import
from __future__ import print_function
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.utils import np_utils
import numpy as np
import unittest

nb_classes = 10
batch_size = 128
nb_epoch = 5
weighted_class = 9
standard_weight = 1
high_weight = 5
max_train_samples = 5000
max_test_samples = 1000

np.random.seed(1337) # for reproducibility

# the data, shuffled and split between tran and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(60000, 784)[:max_train_samples]
X_test = X_test.reshape(10000, 784)[:max_test_samples]
X_train = X_train.astype("float32") / 255
X_test = X_test.astype("float32") / 255

# convert class vectors to binary class matrices
y_train = y_train[:max_train_samples]
y_test = y_test[:max_test_samples]
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)
test_ids = np.where(y_test == np.array(weighted_class))[0]

def create_model():
    model = Sequential()
    model.add(Dense(784, 50))
    model.add(Activation('relu'))
    model.add(Dense(50, 10))
    model.add(Activation('softmax'))
    return model

def test_weights(model, class_weight=None, sample_weight=None):
    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=0, \
        class_weight=class_weight, sample_weight=sample_weight)
    score = model.evaluate(X_test[test_ids, :], Y_test[test_ids, :], verbose=0)
    return score

class TestConcatenation(unittest.TestCase):

    def test_loss_weighting(self):
        class_weight = dict([(i, standard_weight) for i in range(nb_classes)])
        class_weight[weighted_class] = high_weight

        sample_weight = np.ones((y_train.shape[0])) * standard_weight
        sample_weight[y_train == weighted_class] = high_weight

        for loss in ['mae', 'mse', 'categorical_crossentropy']:
            print('loss:', loss)
            # no weights: reference point
            model = create_model()
            model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
            standard_score = test_weights(model)
            # test class_weight
            model = create_model()
            model.compile(loss=loss, optimizer='rmsprop')
            score = test_weights(model, class_weight=class_weight)
            print('score:', score, ' vs.', standard_score)
            self.assertTrue(score < standard_score)
            # test sample_weight
            model = create_model()
            model.compile(loss=loss, optimizer='rmsprop')
            score = test_weights(model, sample_weight=sample_weight)
            print('score:', score, ' vs.', standard_score)
            self.assertTrue(score < standard_score)

if __name__ == '__main__':
    print('Test class_weight and sample_weight')
    unittest.main()