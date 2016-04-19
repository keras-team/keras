from . import backend as K


def binary_accuracy(y_true, y_pred):
    return K.mean(K.equal(y_true, K.round(y_pred)))


def categorical_accuracy(y_true, y_pred):
    return K.mean(K.equal(K.argmax(y_true, axis=-1),
                  K.argmax(y_pred, axis=-1)))


from .utils.generic_utils import get_from_module
def get(identifier):
    return get_from_module(identifier, globals(), 'metric')
