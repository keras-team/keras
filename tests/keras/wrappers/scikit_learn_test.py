import pytest
import numpy as np

from keras.utils.test_utils import get_test_data

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor

input_dim = 5
hidden_dims = 5
num_train = 100
num_test = 50
num_classes = 3
batch_size = 32
epochs = 1
verbosity = 0
optim = 'adam'
loss = 'categorical_crossentropy'

np.random.seed(42)
(X_train, y_train), (X_test, y_test) = get_test_data(
    num_train=num_train, num_test=num_test, input_shape=(input_dim,),
    classification=True, num_classes=num_classes)


def build_fn_clf(input_shape, output_shape, hidden_dims):
    model = Sequential()
    model.add(Activation('relu'))
    model.add(Dense(hidden_dims))
    model.add(Activation('relu'))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    model.compile(optimizer='sgd', loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def test_classify_build_fn():
    clf = KerasClassifier(
        build_fn=build_fn_clf, hidden_dims=hidden_dims,
        batch_size=batch_size, epochs=epochs)

    assert_classification_works(clf)
    assert_string_classification_works(clf)


def test_classify_class_build_fn():
    class ClassBuildFnClf(object):

        def __call__(self, input_shape, output_shape, hidden_dims):
            return build_fn_clf(input_shape, output_shape, hidden_dims)

    clf = KerasClassifier(
        build_fn=ClassBuildFnClf(), hidden_dims=hidden_dims,
        batch_size=batch_size, epochs=epochs)

    assert_classification_works(clf)
    assert_string_classification_works(clf)


def test_classify_inherit_class_build_fn():
    class InheritClassBuildFnClf(KerasClassifier):

        def __call__(self, input_shape, output_shape, hidden_dims):
            return build_fn_clf(input_shape, output_shape, hidden_dims)

    clf = InheritClassBuildFnClf(
        build_fn=None, hidden_dims=hidden_dims,
        batch_size=batch_size, epochs=epochs)

    assert_classification_works(clf)
    assert_string_classification_works(clf)


def assert_classification_works(clf):
    clf.fit(X_train, y_train, sample_weight=np.ones(X_train.shape[0]),
            batch_size=batch_size, epochs=epochs)

    score = clf.score(X_train, y_train, batch_size=batch_size)
    assert np.isscalar(score) and np.isfinite(score)

    preds = clf.predict(X_test, batch_size=batch_size)
    assert preds.shape == (num_test, )
    for prediction in np.unique(preds):
        assert prediction in range(num_classes)

    proba = clf.predict_proba(X_test, batch_size=batch_size)
    assert proba.shape == (num_test, num_classes)
    assert np.allclose(np.sum(proba, axis=1), np.ones(num_test))


def assert_string_classification_works(clf):
    string_classes = ['cls{}'.format(x) for x in range(num_classes)]
    str_y_train = np.array(string_classes)[y_train]

    clf.fit(X_train, str_y_train, batch_size=batch_size, epochs=epochs)

    score = clf.score(X_train, str_y_train, batch_size=batch_size)
    assert np.isscalar(score) and np.isfinite(score)

    preds = clf.predict(X_test, batch_size=batch_size)
    assert preds.shape == (num_test, )
    for prediction in np.unique(preds):
        assert prediction in string_classes

    proba = clf.predict_proba(X_test, batch_size=batch_size)
    assert proba.shape == (num_test, num_classes)
    assert np.allclose(np.sum(proba, axis=1), np.ones(num_test))


def build_fn_reg(input_shape, output_shape, hidden_dims=50):
    model = Sequential()
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
        batch_size=batch_size, epochs=epochs)

    assert_regression_works(reg)


def test_regression_class_build_fn():
    class ClassBuildFnReg(object):

        def __call__(self, input_shape, output_shape, hidden_dims):
            return build_fn_reg(input_shape, output_shape, hidden_dims)

    reg = KerasRegressor(
        build_fn=ClassBuildFnReg(), hidden_dims=hidden_dims,
        batch_size=batch_size, epochs=epochs)

    assert_regression_works(reg)


def test_regression_inherit_class_build_fn():
    class InheritClassBuildFnReg(KerasRegressor):

        def __call__(self, input_shape, output_shape, hidden_dims):
            return build_fn_reg(input_shape, output_shape, hidden_dims)

    reg = InheritClassBuildFnReg(
        build_fn=None, hidden_dims=hidden_dims,
        batch_size=batch_size, epochs=epochs)

    assert_regression_works(reg)


def assert_regression_works(reg):
    reg.fit(X_train, y_train, batch_size=batch_size, epochs=epochs)

    score = reg.score(X_train, y_train, batch_size=batch_size)
    assert np.isscalar(score) and np.isfinite(score)

    preds = reg.predict(X_test, batch_size=batch_size)
    assert preds.shape == (num_test, )


def test_regression_predict_shape_correct_num_test_0():
    assert_regression_predict_shape_correct(num_test=0)


def test_regression_predict_shape_correct_num_test_1():
    assert_regression_predict_shape_correct(num_test=1)


def assert_regression_predict_shape_correct(num_test):
    reg = KerasRegressor(
        build_fn=build_fn_reg, hidden_dims=hidden_dims,
        batch_size=batch_size, epochs=epochs)
    reg.fit(X_train, y_train, batch_size=batch_size, epochs=epochs)

    preds = reg.predict(X_test[:num_test], batch_size=batch_size)
    assert preds.shape == (num_test, )


if __name__ == '__main__':
    pytest.main([__file__])


# Usage of sklearn's Pipelines, SearchCVs, Ensembles and CalibratedClassifierCVs


# from keras import backend as K
# from keras.layers import Conv2D, Dense, Flatten, Input
# from keras.models import Model
# from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
# import numpy as np
# import pickle
# from scipy.stats import randint
# from sklearn.calibration import CalibratedClassifierCV
# from sklearn.datasets import load_boston, load_digits, load_iris
# from sklearn.ensemble import (AdaBoostClassifier, AdaBoostRegressor,
#                               BaggingClassifier, BaggingRegressor)
# from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler


# def load_digits8x8():
#     data = load_digits()
#     data.data = data.data.reshape([data.data.shape[0], 1, 8, 8]) / 16.0
#     K.set_image_data_format('channels_first')
#     return data


# def check(estimator, loader):
#     data = loader()
#     estimator.fit(data.data, data.target)
#     preds = estimator.predict(data.data)
#     score = estimator.score(data.data, data.target)
#     serialized_estimator = pickle.dumps(estimator)
#     K.clear_session()
#     deserialized_estimator = pickle.loads(serialized_estimator)
#     preds = deserialized_estimator.predict(data.data)
#     score = deserialized_estimator.score(data.data, data.target)
#     assert True


# def build_fn_regs(input_shape, output_shape, hidden_layer_sizes=[]):
#     model = Sequential()
#     for size in hidden_layer_sizes:
#         model.add(Dense(size, activation='relu'))
#     model.add(Dense(np.prod(output_shape, dtype=np.uint8)))
#     model.compile('adam', loss='mean_squared_error')
#     return model


# def build_fn_clss(input_shape, output_shape, hidden_layer_sizes=[]):
#     model = Sequential()
#     for size in hidden_layer_sizes:
#         model.add(Dense(size, activation='relu'))
#     model.add(Dense(np.prod(output_shape, dtype=np.uint8),
#                     activation='softmax'))
#     model.compile('adam', loss='categorical_crossentropy', metrics=['accuracy'])
#     return model


# def build_fn_clscs(input_shape, output_shape, hidden_layer_sizes=[]):
#     model = Sequential()
#     model.add(Conv2D(3, (3, 3)))
#     model.add(Flatten())
#     for size in hidden_layer_sizes:
#         model.add(Dense(size, activation='relu'))
#     model.add(Dense(np.prod(output_shape, dtype=np.uint8),
#                     activation='softmax'))
#     model.compile('adam', loss='categorical_crossentropy', metrics=['accuracy'])
#     return model


# def build_fn_clscf(input_shape, output_shape, hidden_layer_sizes=[]):
#     # https://github.com/keras-team/keras/issues/5176
#     x = Input(shape=input_shape)
#     z = Conv2D(3, (3, 3))(x)
#     z = Flatten()(z)
#     for size in hidden_layer_sizes:
#         z = Dense(size, activation='relu')(z)
#     y = Dense(np.prod(output_shape, dtype=np.uint8),
#                     activation='softmax')(z)
#     model = Model(inputs=x, outputs=y)
#     model.compile('adam', loss='categorical_crossentropy', metrics=['accuracy'])
#     return model


# CONFIG = {'MLPRegressor': (load_boston, KerasRegressor, build_fn_regs,
#                            (BaggingRegressor, AdaBoostRegressor)),
#           'MLPClassifier': (load_iris, KerasClassifier, build_fn_clss,
#                             (BaggingClassifier, AdaBoostClassifier)),
#           'CNNClassifier': (load_digits8x8, KerasClassifier, build_fn_clscs,
#                             (BaggingClassifier, AdaBoostClassifier)),
#           'CNNClassifierF': (load_digits8x8, KerasClassifier, build_fn_clscf,
#                              (BaggingClassifier, AdaBoostClassifier))}


# def test_standalone():
#     """Tests standalone estimator."""
#     for config in ['MLPRegressor', 'MLPClassifier', 'CNNClassifier',
#                    'CNNClassifierF']:
#         loader, model, build_fn, _ = CONFIG[config]
#         estimator = model(build_fn, epochs=1)
#         check(estimator, loader)


# def test_pipeline():
#     """Tests compatibility with Scikit-learn's pipeline."""
#     for config in ['MLPRegressor', 'MLPClassifier']:
#         loader, model, build_fn, _ = CONFIG[config]
#         estimator = model(build_fn, epochs=1)
#         estimator = Pipeline([('s', StandardScaler()), ('e', estimator)])
#         check(estimator, loader)


# def test_searchcv():
#     """Tests compatibility with Scikit-learn's hyperparameter search CV."""
#     for config in ['MLPRegressor', 'MLPClassifier', 'CNNClassifier',
#                    'CNNClassifierF']:
#         loader, model, build_fn, _ = CONFIG[config]
#         estimator = model(build_fn, epochs=1, validation_split=0.1)
#         check(GridSearchCV(estimator, {'hidden_layer_sizes': [[], [5]]}),
#               loader)
#         check(RandomizedSearchCV(estimator, {'epochs': randint(1, 5)},
#                                  n_iter=2), loader)


# def test_ensemble():
#     """Tests compatibility with Scikit-learn's ensembles."""
#     for config in ['MLPRegressor', 'MLPClassifier']:
#         loader, model, build_fn, ensembles = CONFIG[config]
#         base_estimator = model(build_fn, epochs=1)
#         for ensemble in ensembles:
#             estimator = ensemble(base_estimator=base_estimator, n_estimators=2)
#             check(estimator, loader)


# def test_calibratedclassifiercv():
#     """Tests compatibility with Scikit-learn's calibrated classifier CV."""
#     for config in ['MLPClassifier']:
#         loader, _, build_fn, _ = CONFIG[config]
#         base_estimator = KerasClassifier(build_fn, epochs=1)
#         estimator = CalibratedClassifierCV(base_estimator=base_estimator)
#         check(estimator, loader)
