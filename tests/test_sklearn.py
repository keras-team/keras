import keras
from keras.layers.core import Dense, Dropout, Activation
from keras.models import Sequential
from keras.optimizers import SGD
from keras.wrappers.scikit_learn import KerasClassifier

import numpy as np

# Data generation
np.random.seed(1337)
X = np.random.rand(200, 10)
X = (X-X.mean(axis=0))/X.std(axis=0)
y = np.random.binomial(1, 1.0/5, size=(200,))


def get_model(outputs):
    model = Sequential()
    model.add(Dense(25, input_dim=10, init='uniform'))
    model.add(Activation('tanh'))
    model.add(Dense(outputs, init='uniform'))
    model.add(Activation('sigmoid'))
    return model

def get_wrapped_model(outputs, loss, epochs):
    model = get_model(outputs)
    rmsprop = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-6)
    clf = KerasClassifier(model,
                          optimizer=rmsprop, 
                          nb_epoch=epochs,
                          class_weight='balanced',
                          loss=loss,
                          verbose=0)
    return clf

    
def test_fit():
    clf = get_wrapped_model(outputs=2, loss='categorical_crossentropy', epochs=2)
    clf.fit(X, y)


def test_score_binary():
    # We will test the score function, and make sure the classes_ are not being changed
    clf = get_wrapped_model(outputs=1, loss='binary_crossentropy', epochs=2)
    clf.fit(X, y)
    score = clf.score(X, y)
    assert len(clf.classes_) == 2
    score = clf.score(X, np.ones(X.shape[0])*y[0])
    assert len(clf.classes_) == 2
    

def test_score_categorical():
    # We will test the score function, and make sure the classes_ are not being changed
    clf = get_wrapped_model(outputs=2, loss='categorical_crossentropy', epochs=2)
    clf.fit(X, y)
    score = clf.score(X, y)
    assert len(clf.classes_) == 2
