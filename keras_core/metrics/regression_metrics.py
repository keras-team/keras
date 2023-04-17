from keras_core import operations as ops
from keras_core.metrics import reduction_metrics


def mean_square_error(y_true, y_pred):
    ndim = len(y_pred.shape)
    return ops.mean((y_true - y_pred) ** 2, axis=list(range(1, ndim)))


class MeanSquareError(reduction_metrics.MeanMetricWrapper):
    def __init__(self, name="mean_square_error", dtype=None):
        super().__init__(fn=mean_square_error, name=name, dtype=dtype)

    def get_config(self):
        return {"name": self.name, "dtype": self.dtype}
