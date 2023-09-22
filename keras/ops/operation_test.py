import numpy as np

from keras import backend
from keras import testing
from keras.backend.common import keras_tensor
from keras.ops import numpy as knp
from keras.ops import operation


class OpWithMultipleInputs(operation.Operation):
    def call(self, x, y, z=None):
        # `z` has to be put first due to the order of operations issue with
        # torch backend.
        return 3 * z + x + 2 * y

    def compute_output_spec(self, x, y, z=None):
        return keras_tensor.KerasTensor(x.shape, x.dtype)


class OpWithMultipleOutputs(operation.Operation):
    def call(self, x):
        return (x, x + 1)

    def compute_output_spec(self, x):
        return (
            keras_tensor.KerasTensor(x.shape, x.dtype),
            keras_tensor.KerasTensor(x.shape, x.dtype),
        )


class OpWithCustomConstructor(operation.Operation):
    def __init__(self, alpha, mode="foo"):
        super().__init__()
        self.alpha = alpha
        self.mode = mode

    def call(self, x):
        if self.mode == "foo":
            return x
        return self.alpha * x

    def compute_output_spec(self, x):
        return keras_tensor.KerasTensor(x.shape, x.dtype)


class OperationTest(testing.TestCase):
    def test_symbolic_call(self):
        x = keras_tensor.KerasTensor(shape=(2, 3), name="x")
        y = keras_tensor.KerasTensor(shape=(2, 3), name="y")
        z = keras_tensor.KerasTensor(shape=(2, 3), name="z")

        # Positional arguments
        op = OpWithMultipleInputs(name="test_op")
        self.assertEqual(op.name, "test_op")
        out = op(x, y, z)
        self.assertIsInstance(out, keras_tensor.KerasTensor)
        self.assertEqual(out.shape, (2, 3))
        self.assertEqual(len(op._inbound_nodes), 1)
        self.assertEqual(op.input, [x, y, z])
        self.assertEqual(op.output, out)

        # Keyword arguments
        op = OpWithMultipleInputs(name="test_op")
        out = op(x=x, y=y, z=z)
        self.assertIsInstance(out, keras_tensor.KerasTensor)
        self.assertEqual(out.shape, (2, 3))
        self.assertEqual(len(op._inbound_nodes), 1)
        self.assertEqual(op.input, [x, y, z])
        self.assertEqual(op.output, out)

        # Mix
        op = OpWithMultipleInputs(name="test_op")
        out = op(x, y=y, z=z)
        self.assertIsInstance(out, keras_tensor.KerasTensor)
        self.assertEqual(out.shape, (2, 3))
        self.assertEqual(len(op._inbound_nodes), 1)
        self.assertEqual(op.input, [x, y, z])
        self.assertEqual(op.output, out)

        # Test op reuse
        prev_out = out
        out = op(x, y=y, z=z)
        self.assertIsInstance(out, keras_tensor.KerasTensor)
        self.assertEqual(out.shape, (2, 3))
        self.assertEqual(len(op._inbound_nodes), 2)
        self.assertEqual(op.output, prev_out)

        # Test multiple outputs
        op = OpWithMultipleOutputs()
        out = op(x)
        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), 2)
        self.assertIsInstance(out[0], keras_tensor.KerasTensor)
        self.assertIsInstance(out[1], keras_tensor.KerasTensor)
        self.assertEqual(out[0].shape, (2, 3))
        self.assertEqual(out[1].shape, (2, 3))
        self.assertEqual(len(op._inbound_nodes), 1)
        self.assertEqual(op.output, list(out))

    def test_eager_call(self):
        x = knp.ones((2, 3))
        y = knp.ones((2, 3))
        z = knp.ones((2, 3))
        op = OpWithMultipleInputs(name="test_op")
        self.assertEqual(op.name, "test_op")

        # Positional arguments
        out = op(x, y, z)
        self.assertTrue(backend.is_tensor(out))
        self.assertAllClose(out, 6 * np.ones((2, 3)))

        # Keyword arguments
        out = op(x=x, y=y, z=z)
        self.assertTrue(backend.is_tensor(out))
        self.assertAllClose(out, 6 * np.ones((2, 3)))

        # Mixed arguments
        out = op(x, y=y, z=z)
        self.assertTrue(backend.is_tensor(out))
        self.assertAllClose(out, 6 * np.ones((2, 3)))

        # Test multiple outputs
        op = OpWithMultipleOutputs()
        out = op(x)
        self.assertEqual(len(out), 2)
        self.assertTrue(backend.is_tensor(out[0]))
        self.assertTrue(backend.is_tensor(out[1]))
        self.assertAllClose(out[0], np.ones((2, 3)))
        self.assertAllClose(out[1], np.ones((2, 3)) + 1)

    def test_serialization(self):
        op = OpWithMultipleOutputs(name="test_op")
        config = op.get_config()
        self.assertEqual(config, {"name": "test_op"})
        op = OpWithMultipleOutputs.from_config(config)
        self.assertEqual(op.name, "test_op")

    def test_autoconfig(self):
        op = OpWithCustomConstructor(alpha=0.2, mode="bar")
        config = op.get_config()
        self.assertEqual(config, {"alpha": 0.2, "mode": "bar"})
        revived = OpWithCustomConstructor.from_config(config)
        self.assertEqual(revived.get_config(), config)

    def test_input_conversion(self):
        x = np.ones((2,))
        y = np.ones((2,))
        z = knp.ones((2,))  # mix
        if backend.backend() == "torch":
            z = z.cpu()
        op = OpWithMultipleInputs()
        out = op(x, y, z)
        self.assertTrue(backend.is_tensor(out))
        self.assertAllClose(out, 6 * np.ones((2,)))

    def test_valid_naming(self):
        OpWithMultipleOutputs(name="test_op")

        with self.assertRaisesRegex(
            ValueError, "must be a string and cannot contain character `/`."
        ):
            OpWithMultipleOutputs(name="test/op")
