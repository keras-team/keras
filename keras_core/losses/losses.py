from keras_core.losses.loss import Loss
from keras_core import operations as ops
from keras_core.losses.loss import squeeze_to_same_rank


class LossFunctionWrapper(Loss):
    def __init__(self, fn, reduction="sum_over_batch_size", name=None, **kwargs):
        super().__init__(reduction=reduction, name=name)
        self.fn = fn
        self._fn_kwargs = kwargs

    def call(self, y_true, y_pred):
        y_true, y_pred = squeeze_to_same_rank(y_true, y_pred)
        return self.fn(y_true, y_pred, **self._fn_kwargs)

    def get_config(self):
        raise NotImplementedError

    @classmethod
    def from_config(clf, config):
        raise NotImplementedError


def mean_squared_error(y_true, y_pred):
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = ops.convert_to_tensor(y_true, dtype=y_pred.dtype)
    y_true, y_pred = squeeze_to_same_rank(y_true, y_pred)
    return (y_true - y_pred) ** 2


class MeanSquaredError(LossFunctionWrapper):
    def __init__(self, reduction="sum_over_batch_size", name="mean_squared_error"):
        super().__init__(mean_squared_error, reduction=reduction, name=name)
