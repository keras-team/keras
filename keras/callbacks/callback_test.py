import numpy as np
import pytest

from keras import models
from keras import testing
from keras.callbacks.callback import Callback


class CallbackTest(testing.TestCase):
    @pytest.mark.requires_trainable_backend
    def test_model_state_is_current_on_epoch_end(self):
        class TestModel(models.Model):
            def __init__(self):
                super().__init__()
                self.iterations = self.add_variable(
                    shape=(), initializer="zeros", trainable=False
                )

            def call(self, inputs):
                self.iterations.assign(self.iterations + 1)
                return inputs

        class CBK(Callback):
            def on_batch_end(self, batch, logs):
                assert np.int32(self.model.iterations) == batch + 1

        model = TestModel()
        model.compile(optimizer="sgd", loss="mse")
        x = np.random.random((8, 1))
        y = np.random.random((8, 1))
        model.fit(x, y, callbacks=[CBK()], batch_size=2)
