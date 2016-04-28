import pytest
import numpy as np

from keras.utils.test_utils import get_test_data
from keras.utils import np_utils
from keras import backend as K

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor

np.random.seed(1337)

input_dim = 10
nb_class = 3
batch_size = 32
nb_epoch = 1
verbosity = 0
optim = 'adam'
loss = 'categorical_crossentropy'


(X_train, y_train), (X_test, y_test) = get_test_data(nb_train=400,
                                                     nb_test=200,
                                                     input_shape=(input_dim,),
                                                     classification=True,
                                                     nb_class=nb_class)
y_train = np_utils.to_categorical(y_train, nb_classes=nb_class)
y_test = np_utils.to_categorical(y_test, nb_classes=nb_class)


(X_train_reg, y_train_reg), (X_test_reg, y_test_reg) = get_test_data(nb_train=400,
                                                                     nb_test=200,
                                                                     input_shape=(input_dim,),
                                                                     classification=False,
                                                                     nb_class=1,
                                                                     output_shape=(1,))


def build_fn_clf(hidden_dims=50):
    model = Sequential()
    model.add(Dense(input_dim, input_shape=(input_dim,)))
    model.add(Activation('relu'))
    model.add(Dense(hidden_dims))
    model.add(Activation('relu'))
    model.add(Dense(nb_class))
    model.add(Activation('softmax'))
    model.compile(optimizer='sgd', loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


class Class_build_fn_clf(object):
    def __call__(self, hidden_dims):
        return build_fn_clf(hidden_dims)


class Inherit_class_build_fn_clf(KerasClassifier):
    def __call__(self, hidden_dims):
        return build_fn_clf(hidden_dims)


def build_fn_reg(hidden_dims=50):
    model = Sequential()
    model.add(Dense(input_dim, input_shape=(input_dim,)))
    model.add(Activation('relu'))
    model.add(Dense(hidden_dims))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.add(Activation('linear'))
    model.compile(optimizer='sgd', loss='mean_absolute_error',
                  metrics=['accuracy'])
    return model


class Class_build_fn_reg(object):
    def __call__(self, hidden_dims):
        return build_fn_reg(hidden_dims)


class Inherit_class_build_fn_reg(KerasRegressor):
    def __call__(self, hidden_dims):
        return build_fn_reg(hidden_dims)

for fn in [build_fn_clf, Class_build_fn_clf(), Inherit_class_build_fn_clf]:
    if fn is Inherit_class_build_fn_clf:
        classifier = Inherit_class_build_fn_clf(
            build_fn=None, hidden_dims=50, batch_size=batch_size, nb_epoch=nb_epoch)
    else:
        classifier = KerasClassifier(
            build_fn=fn, hidden_dims=50, batch_size=batch_size, nb_epoch=nb_epoch)

    classifier.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch)
    score = classifier.score(X_train, y_train, batch_size=batch_size)
    preds = classifier.predict(X_test, batch_size=batch_size)
    proba = classifier.predict_proba(X_test, batch_size=batch_size)


for fn in [build_fn_reg, Class_build_fn_reg(), Inherit_class_build_fn_reg]:
    if fn is Inherit_class_build_fn_reg:
        regressor = Inherit_class_build_fn_reg(
            build_fn=None, hidden_dims=50, batch_size=batch_size, nb_epoch=nb_epoch)
    else:
        regressor = KerasRegressor(
            build_fn=fn, hidden_dims=50, batch_size=batch_size, nb_epoch=nb_epoch)

    regressor.fit(X_train_reg, y_train_reg,
                  batch_size=batch_size, nb_epoch=nb_epoch)
    score = regressor.score(X_train_reg, y_train_reg, batch_size=batch_size)
    preds = regressor.predict(X_test, batch_size=batch_size)


# Usage of sklearn's grid_search
# from sklearn import grid_search
# parameters = dict(hidden_dims = [20, 30], batch_size=[64, 128], nb_epoch=[2], verbose=[0])
# classifier = Inherit_class_build_fn_clf()
# clf = grid_search.GridSearchCV(classifier, parameters)
# clf.fit(X_train, y_train)
# parameters = dict(hidden_dims = [20, 30], batch_size=[64, 128], nb_epoch=[2], verbose=[0])
# regressor = Inherit_class_build_fn_reg()
# reg = grid_search.GridSearchCV(regressor, parameters, scoring='mean_squared_error', n_jobs=1, cv=2, verbose=2)
# reg.fit(X_train_reg, y_train_reg)
