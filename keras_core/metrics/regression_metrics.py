from keras_core import backend
from keras_core import initializers
from keras_core.metrics.metric import Metric


class MeanSquareError(Metric):
    def __init__(self, name="mean_square_error", dtype=None):
        super().__init__(name=name, dtype=dtype)
        self.sum = self.add_variable(
            name="sum", initializer=initializers.Zeros()
        )
        self.total = self.add_variable(
            name="total", initializer=initializers.Zeros()
        )

    def update_state(self, y_true, y_pred):
        # TODO: add support for sample_weight
        sum = (y_true - y_pred) ** 2
        self.sum.assign(self.sum + sum)
        batch_size = backend.shape(y_true)[0]
        self.total.assign(self.total + batch_size)

    def result(self):
        return self.sum / self.total

    def reset_states(self):
        self.sum.assign(0.0)
        self.total.assign(0.0)
