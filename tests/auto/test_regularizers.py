import unittest
import numpy as np
np.random.seed(1337) # for reproducibility

from keras.models import Sequential
from keras.layers.core import Merge, Dense, Activation, Flatten, ActivityRegularization
from keras.layers.embeddings import Embedding
from keras.datasets import mnist
from keras.utils import np_utils
from keras import regularizers

nb_classes = 10
batch_size = 128
nb_epoch = 5
weighted_class = 9
standard_weight = 1
high_weight = 5
max_train_samples = 5000
max_test_samples = 1000

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

def create_model(weight_reg=None, activity_reg=None):
    model = Sequential()
    model.add(Dense(784, 50))
    model.add(Activation('relu'))
    model.add(ActivityRegularization(activity_reg))
    model.add(Dense(50, 10, W_regularizer=weight_reg))
    model.add(Activation('softmax'))
    return model


class TestRegularizers(unittest.TestCase):
    def test_W_reg(self):
        for reg in [regularizers.identity(), regularizers.l1(), regularizers.l2(), regularizers.l1l2()]:
            model = create_model(weight_reg=reg)
            model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
            model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=0)
            model.evaluate(X_test[test_ids, :], Y_test[test_ids, :], verbose=0)

    def test_A_reg(self):
        for reg in [regularizers.activity_l1(), regularizers.activity_l2()]:
            model = create_model(activity_reg=reg)
            model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
            model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=0)
            model.evaluate(X_test[test_ids, :], Y_test[test_ids, :], verbose=0)

if __name__ == '__main__':
    print('Test weight and activity regularizers')
    unittest.main()
