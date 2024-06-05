import json

import numpy as np

from keras.src import testing
from keras.src.backend.common import keras_tensor
from keras.src.layers import Dense
from keras.src.layers import Input
from keras.src.models import Model
from keras.src.ops import function
from keras.src.ops import numpy as knp


class FunctionTest(testing.TestCase):
    def test_define_and_call(self):
        x1 = keras_tensor.KerasTensor((2, 3))
        x2 = keras_tensor.KerasTensor((2, 3))
        x = knp.add(x1, x2)
        y1 = x * 3
        y2 = x**2
        fn = function.Function(
            inputs=[x1, x2], outputs=[y1, y2], name="test_function"
        )
        self.assertEqual(fn.name, "test_function")

        # Eager call
        y_val = fn([np.ones((2, 3)), np.ones((2, 3))])
        self.assertIsInstance(y_val, list)
        self.assertAllClose(y_val[0], np.ones((2, 3)) * 6)
        self.assertAllClose(y_val[1], np.ones((2, 3)) * 4)

        # Symbolic call
        x1_alt = keras_tensor.KerasTensor((2, 3))
        x2_alt = keras_tensor.KerasTensor((2, 3))
        y_val = fn([x1_alt, x2_alt])
        self.assertIsInstance(y_val[0], keras_tensor.KerasTensor)
        self.assertEqual(y_val[0].shape, (2, 3))
        self.assertIsInstance(y_val[1], keras_tensor.KerasTensor)
        self.assertEqual(y_val[1].shape, (2, 3))

        # Recursion
        fn = function.Function(inputs=[x1_alt, x2_alt], outputs=y_val)
        y_val = fn([np.ones((2, 3)), np.ones((2, 3))])
        self.assertIsInstance(y_val, list)
        self.assertAllClose(y_val[0], np.ones((2, 3)) * 6)
        self.assertAllClose(y_val[1], np.ones((2, 3)) * 4)

    def test_dynamic_shape_inference(self):
        x = keras_tensor.KerasTensor((None, 3))
        y = x**2
        fn = function.Function(x, y)

        # Test with compute_output_spec
        out = fn.compute_output_spec(keras_tensor.KerasTensor((4, 3)))
        self.assertIsInstance(out, keras_tensor.KerasTensor)
        self.assertEqual(out.shape, (4, 3))

        # Test with compute_output_shape
        out = fn.compute_output_shape((None, 3))
        self.assertIsInstance(out, tuple)
        self.assertEqual(out, (None, 3))

        # Test with call
        out = fn(keras_tensor.KerasTensor((4, 3)))
        self.assertIsInstance(out, keras_tensor.KerasTensor)
        self.assertEqual(out.shape, (4, 3))

    def test_dict_io(self):
        x1 = keras_tensor.KerasTensor((2, 3))
        x2 = keras_tensor.KerasTensor((2, 3))
        x = knp.add(x1, x2)
        y1 = x * 3
        y2 = x**2
        fn = function.Function(
            inputs={"x1": x1, "x2": x2}, outputs={"y1": y1, "y2": y2}
        )

        # Eager call
        y_val = fn({"x1": np.ones((2, 3)), "x2": np.ones((2, 3))})
        self.assertIsInstance(y_val, dict)
        self.assertAllClose(y_val["y1"], np.ones((2, 3)) * 6)
        self.assertAllClose(y_val["y2"], np.ones((2, 3)) * 4)

        # Symbolic call
        x1_alt = keras_tensor.KerasTensor((2, 3))
        x2_alt = keras_tensor.KerasTensor((2, 3))
        y_val = fn({"x1": x1_alt, "x2": x2_alt})
        self.assertIsInstance(y_val["y1"], keras_tensor.KerasTensor)
        self.assertEqual(y_val["y1"].shape, (2, 3))
        self.assertIsInstance(y_val["y2"], keras_tensor.KerasTensor)
        self.assertEqual(y_val["y2"].shape, (2, 3))

    def test_invalid_inputs_error(self):
        x1 = keras_tensor.KerasTensor((2, 3))
        x2 = keras_tensor.KerasTensor((2, 3))
        x = knp.add(x1, x2)
        y1 = x * 3
        y2 = x**2
        fn = function.Function(
            inputs=[x1, x2], outputs=[y1, y2], name="test_function"
        )
        self.assertEqual(fn.name, "test_function")

        # Bad structure
        with self.assertRaisesRegex(ValueError, "invalid input structure"):
            _ = fn(np.ones((2, 3)))

        # Bad rank
        with self.assertRaisesRegex(ValueError, "incompatible inputs"):
            _ = fn([np.ones((2, 3, 3)), np.ones((2, 3))])

        # Bad shape
        with self.assertRaisesRegex(ValueError, "incompatible inputs"):
            _ = fn([np.ones((4, 3)), np.ones((2, 3))])

    def test_graph_disconnected_error(self):
        # TODO
        pass

    def test_serialization(self):
        inputs = Input(shape=(10,))
        outputs = Dense(1)(inputs)
        model = Model(inputs=inputs, outputs=outputs)

        config = model.get_config()
        new_model = Model.from_config(config)

        self.assertEqual(
            json.dumps(model.get_config()), json.dumps(new_model.get_config())
        )

    def test_function_with_empty_outputs(self):
        x = keras_tensor.KerasTensor((None, 3))
        with self.assertRaisesRegex(
            ValueError, "`outputs` argument cannot be empty"
        ):
            _ = function.Function(inputs=x, outputs=[])

    def test_function_with_empty_inputs(self):
        x = keras_tensor.KerasTensor((None, 3))
        with self.assertRaisesRegex(
            ValueError, "`inputs` argument cannot be empty"
        ):
            _ = function.Function(inputs=[], outputs=x)
