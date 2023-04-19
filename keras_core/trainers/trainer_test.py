import numpy as np

from keras_core import backend
from keras_core import initializers
from keras_core import layers
from keras_core import losses
from keras_core import metrics
from keras_core import optimizers
from keras_core import testing

if backend.backend() == "jax":
    from keras_core.backend.jax.trainer import Trainer
else:
    from keras_core.backend.tensorflow.trainer import Trainer


# A model is just a layer mixed in with a Trainer.
class ExampleModel(layers.Dense, Trainer):
    def __init__(self, units):
        layers.Dense.__init__(
            self,
            units=units,
            use_bias=False,
            kernel_initializer=initializers.Ones(),
        )
        Trainer.__init__(self)


class TestTrainer(testing.TestCase):
    def _test_fit_flow(self, run_eagerly, jit_compile):
        model = ExampleModel(units=3)
        x = np.ones((100, 4))
        y = np.zeros((100, 3))
        batch_size = 16
        epochs = 3

        model.compile(
            optimizer=optimizers.SGD(),
            loss=losses.MeanSquaredError(),
            metrics=[metrics.MeanSquaredError()],
            run_eagerly=run_eagerly,
            jit_compile=jit_compile,
        )
        history = model.fit(x, y, batch_size=batch_size, epochs=epochs)
        history = history.history
        self.assertIn("loss", history)
        self.assertIn("mean_squared_error", history)
        self.assertAllClose(
            history["mean_squared_error"], [13.938, 9.547, 6.539], atol=1e-2
        )

    def test_fit_flow_eager(self):
        self._test_fit_flow(run_eagerly=True, jit_compile=False)

    def test_fit_flow_graph_fn(self):
        self._test_fit_flow(run_eagerly=False, jit_compile=False)

    def test_fit_flow_jit(self):
        self._test_fit_flow(run_eagerly=False, jit_compile=True)

    def _test_evaluate_flow(self, run_eagerly, jit_compile):
        model = ExampleModel(units=3)
        x = np.ones((100, 4))
        y = np.zeros((100, 3))
        batch_size = 16

        model.compile(
            optimizer=optimizers.SGD(),
            loss=losses.MeanSquaredError(),
            metrics=[metrics.MeanSquaredError()],
            run_eagerly=run_eagerly,
            jit_compile=jit_compile,
        )
        output = model.evaluate(x, y, batch_size=batch_size)
        self.assertAllClose(output, [16.0, 16.0])
        output = model.evaluate(x, y, batch_size=batch_size, return_dict=True)
        self.assertTrue(isinstance(output, dict))
        self.assertIn("loss", output)
        self.assertIn("mean_squared_error", output)
        self.assertAllClose(output["mean_squared_error"], 16.0)

    def test_evaluate_flow_eager(self):
        self._test_evaluate_flow(run_eagerly=True, jit_compile=False)

    def test_evaluate_flow_graph_fn(self):
        self._test_evaluate_flow(run_eagerly=False, jit_compile=False)

    def test_evaluate_flow_jit(self):
        self._test_evaluate_flow(run_eagerly=False, jit_compile=True)
