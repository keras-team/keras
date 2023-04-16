from keras_core import backend
from keras_core import initializers
from keras_core import operations as ops
from keras_core.metrics import reduction_metrics
from keras_core.metrics.metric import Metric

# class MeanSquareError(Metric):
#     def __init__(self, name="mean_square_error", dtype=None):
#         super().__init__(name=name, dtype=dtype)
#         self.sum = self.add_variable(
#             name="sum", shape=(), initializer=initializers.Zeros()
#         )
#         self.total = self.add_variable(
#             name="total", shape=(), initializer=initializers.Zeros()
#         )

#     def update_state(self, y_true, y_pred, sample_weight=None):
#         # TODO: add support for sample_weight
#         batch_sum = ops.cast(ops.sum((y_true - y_pred) ** 2), dtype=self.dtype)
#         self.sum.assign(self.sum + batch_sum)
#         batch_size = backend.shape(y_true)[0]
#         self.total.assign(self.total + ops.cast(batch_size, dtype=self.total.dtype))

#     def result(self):
#         return self.sum / (self.total + backend.epsilon())

#     def reset_states(self):
#         self.sum.assign(0.0)
#         self.total.assign(0.0)


def mean_square_error(y_true, y_pred):
    ndim = len(y_pred.shape)
    return ops.mean((y_true - y_pred) ** 2, axis=list(range(1, ndim)))


class MeanSquareError(reduction_metrics.MeanMetricWrapper):
    def __init__(self, name="mean_square_error", dtype=None):
        super().__init__(fn=mean_square_error, name=name, dtype=dtype)

    def get_config(self):
        return {"name": self.name, "dtype": self.dtype}
