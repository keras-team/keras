from __future__ import absolute_import
from . import backend as K


def accuracy(y_true, y_pred, class_mode):
    if class_mode == "categorical":
        if y_true.ndim == 2:
            y_true = K.argmax(y_true, axis=-1)
        return K.mean(K.equal(K.argmax(y_pred, axis=-1), y_true))
    elif class_mode == "binary":
        return K.mean(K.equal(y_pred, K.round(y_true)))
    else:
        raise Exception("Invalid class mode:" + str(class_mode))

# aliases
acc = ACC = accuracy

from .utils.generic_utils import get_from_module
def get(identifier):
    return get_from_module(identifier, globals(), 'metric')
