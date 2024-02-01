import numpy as np
from absl.testing import parameterized

import keras
from keras.layers import Dense
from keras.testing import test_case
from keras.testing import test_utils


class TestAssertNotAllClose(test_case.TestCase):

    def test_assertNotAllClose_with_close_values(self):
        array1 = np.array([1.0, 2.0, 3.0])
        array2 = np.array([1.0, 2.0, 3.0000001])

        with self.assertRaisesRegex(
            AssertionError, "The two values are close at all elements"
        ):
            self.assertNotAllClose(array1, array2)


class TestRunLayerTestErrors(test_case.TestCase):

    def test_run_layer_test_input_shape_and_data_error(self):
        with self.assertRaisesRegex(
            ValueError,
            "input_shape and input_data cannot be passed at the same time",
        ):
            self.run_layer_test(
                layer_cls=Dense,
                init_kwargs={"units": 10},
                input_shape=(10, 10),
                input_data=np.random.random((10, 10)),
            )

    def test_run_layer_test_expected_output_shape_and_output_error(self):
        with self.assertRaisesRegex(
            ValueError,
            "expected_output_shape and expected_output cannot be passed at the same time",
        ):
            self.run_layer_test(
                layer_cls=Dense,
                init_kwargs={"units": 10},
                expected_output_shape=(10, 10),
                expected_output=np.random.random((10, 10)),
            )

    def test_run_layer_test_expected_output_without_input_data_error(self):
        with self.assertRaisesRegex(
            ValueError,
            "In order to use expected_output, input_data must be provided",
        ):
            self.run_layer_test(
                layer_cls=Dense,
                init_kwargs={"units": 10},
                expected_output=np.random.random((10, 10)),
            )

    def test_run_layer_test_expected_mask_shape_without_support_masking_error(
        self,
    ):
        with self.assertRaisesRegex(
            ValueError,
            "In order to use expected_mask_shape, supports_masking\s+must be True",
        ):
            self.run_layer_test(
                layer_cls=Dense,
                init_kwargs={"units": 10},
                expected_mask_shape=(10, 10),
                supports_masking=False,
            )


class TestRunOutputAsserts(test_case.TestCase):

    def test_run_output_asserts_with_tuple_shape(self):
        layer = Dense(units=10)
        input_data = np.random.random((5, 10))
        output = layer(input_data)
        self.run_output_asserts(layer, output, expected_output_shape=(5, 10))

    def test_run_output_asserts_with_dict_shape(self):
        inputs = keras.Input(shape=(10,))
        output1 = Dense(units=10, name="output1")(inputs)
        output2 = Dense(units=5, name="output2")(inputs)
        model = keras.Model(
            inputs=inputs, outputs={"output1": output1, "output2": output2}
        )
        input_data = np.random.random((5, 10))
        output = model(input_data)
        expected_shape = {"output1": (5, 10), "output2": (5, 5)}
        self.run_output_asserts(
            model, output, expected_output_shape=expected_shape
        )

    def test_run_output_asserts_with_list_shape(self):
        inputs = keras.Input(shape=(10,))
        output1 = Dense(units=10)(inputs)
        output2 = Dense(units=5)(inputs)
        model = keras.Model(inputs=inputs, outputs=[output1, output2])
        input_data = np.random.random((5, 10))
        output = model(input_data)
        expected_shape = [(5, 10), (5, 5)]
        self.run_output_asserts(
            model, output, expected_output_shape=expected_shape
        )
