import pytest
import numpy as np

from keras.utils.test_utils import get_test_data
from keras.utils import np_utils
from keras import backend as K

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor

input_dim = 5
hidden_dims = 5
nb_train = 100
nb_test = 50
nb_class = 3
batch_size = 32
nb_epoch = 1
verbosity = 0
optim = 'adam'
loss = 'categorical_crossentropy'

np.random.seed(42)
(X_train, y_train), (X_test, y_test) = get_test_data(
    nb_train=nb_train, nb_test=nb_test, input_shape=(input_dim,),
    classification=True, nb_class=nb_class)


def build_fn_clf(hidden_dims):
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


def test_clasify_build_fn():
    clf = KerasClassifier(
        build_fn=build_fn_clf, hidden_dims=hidden_dims,
        batch_size=batch_size, nb_epoch=nb_epoch)

    assert_classification_works(clf)


def test_clasify_class_build_fn():
    class ClassBuildFnClf(object):

        def __call__(self, hidden_dims):
            return build_fn_clf(hidden_dims)

    clf = KerasClassifier(
        build_fn=ClassBuildFnClf(), hidden_dims=hidden_dims,
        batch_size=batch_size, nb_epoch=nb_epoch)

    assert_classification_works(clf)


def test_clasify_inherit_class_build_fn():
    class InheritClassBuildFnClf(KerasClassifier):

        def __call__(self, hidden_dims):
            return build_fn_clf(hidden_dims)

    clf = InheritClassBuildFnClf(
        build_fn=None, hidden_dims=hidden_dims,
        batch_size=batch_size, nb_epoch=nb_epoch)

    assert_classification_works(clf)


def assert_classification_works(clf):
    clf.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch)

    score = clf.score(X_train, y_train, batch_size=batch_size)
    assert np.isscalar(score) and np.isfinite(score)

    preds = clf.predict(X_test, batch_size=batch_size)
    assert preds.shape == (nb_test, )
    for prediction in np.unique(preds):
        assert prediction in range(nb_class)

    proba = clf.predict_proba(X_test, batch_size=batch_size)
    assert proba.shape == (nb_test, nb_class)
    assert np.allclose(np.sum(proba, axis=1), np.ones(nb_test))


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


def test_regression_build_fn():
    reg = KerasRegressor(
        build_fn=build_fn_reg, hidden_dims=hidden_dims,
        batch_size=batch_size, nb_epoch=nb_epoch)

    assert_regression_works(reg)


def test_regression_class_build_fn():
    class ClassBuildFnReg(object):

        def __call__(self, hidden_dims):
            return build_fn_reg(hidden_dims)

    reg = KerasRegressor(
        build_fn=ClassBuildFnReg(), hidden_dims=hidden_dims,
        batch_size=batch_size, nb_epoch=nb_epoch)

    assert_regression_works(reg)


def test_regression_inherit_class_build_fn():
    class InheritClassBuildFnReg(KerasRegressor):

        def __call__(self, hidden_dims):
            return build_fn_reg(hidden_dims)

    reg = InheritClassBuildFnReg(
        build_fn=None, hidden_dims=hidden_dims,
        batch_size=batch_size, nb_epoch=nb_epoch)

    assert_regression_works(reg)


def assert_regression_works(reg):
    reg.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch)

    score = reg.score(X_train, y_train, batch_size=batch_size)
    assert np.isscalar(score) and np.isfinite(score)

    preds = reg.predict(X_test, batch_size=batch_size)
    assert preds.shape == (nb_test, )


if __name__ == '__main__':
    pytest.main([__file__])

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
