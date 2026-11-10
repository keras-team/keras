"""Tests for `export_xla_saved_model` and `model.export("xla_saved_model")`.

The `XLASavedModelArchive` itself is covered in
`xla_saved_model_archive_test.py`; these tests cover the high-level export
entry points.
"""

import os

import numpy as np
import pytest
from absl.testing import parameterized

from keras.src import backend
from keras.src import layers
from keras.src import models
from keras.src import ops
from keras.src import random
from keras.src import testing
from keras.src.backend import config
from keras.src.export import xla_saved_model
from keras.src.export import xla_saved_model_archive_test
from keras.src.testing.test_utils import named_product
from keras.src.utils.module_utils import tensorflow as tf
from keras.src.utils.module_utils import xla_saved_model as xsm_module

get_model = xla_saved_model_archive_test.get_model
assert_signature = xla_saved_model_archive_test.assert_signature
exported_function_names = xla_saved_model_archive_test.exported_function_names


@pytest.mark.skipif(
    config.backend() != "jax",
    reason="`export_xla_saved_model` only supports the JAX backend.",
)
@pytest.mark.skipif(
    not xsm_module.available,
    reason="Requires the `xla_saved_model` package.",
)
class ExportXLASavedModelTest(testing.TestCase):
    @parameterized.named_parameters(
        named_product(model_type=["sequential", "functional", "subclass"])
    )
    def test_standard_model_export(self, model_type):
        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model")
        model = get_model(model_type)
        ref_input = np.random.normal(size=(3, 10)).astype("float32")
        ref_output = model(ref_input)

        # The input signature is inferred from the model.
        xla_saved_model.export_xla_saved_model(
            model, temp_filepath, platforms=["cpu"]
        )
        self.assertTrue(
            os.path.isfile(os.path.join(temp_filepath, "manifest.pb"))
        )
        self.assertTrue(
            os.path.isdir(os.path.join(temp_filepath, "checkpoint"))
        )
        # The forward pass is registered as "serving_default", the signature
        # the XLA Serving Engine uses when a request does not specify one.
        self.assertEqual(
            exported_function_names(temp_filepath), ["serving_default"]
        )
        assert_signature(
            self,
            temp_filepath,
            "serving_default",
            {"inputs": ref_input},
            {"output_0": ref_output},
        )
        # Test with a different batch size
        other_input = np.random.normal(size=(6, 10)).astype("float32")
        assert_signature(
            self,
            temp_filepath,
            "serving_default",
            {"inputs": other_input},
            {"output_0": model(other_input)},
        )

    @parameterized.named_parameters(
        named_product(
            model_type=["sequential", "functional", "subclass"],
            input_signature=[
                layers.InputSpec(
                    dtype="float32", shape=(None, 10), name="inputs"
                ),
                "tf_tensor_spec",
                backend.KerasTensor((None, 10), dtype="float32", name="inputs"),
                "backend_tensor",
            ],
        )
    )
    def test_input_signature(self, model_type, input_signature):
        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model")
        model = get_model(model_type)
        ref_input = random.normal((3, 10))
        ref_output = model(ref_input)

        if input_signature == "backend_tensor":
            input_signature = (ref_input,)
        elif input_signature == "tf_tensor_spec":
            if not tf.available:
                self.skipTest("Requires TensorFlow for `tf.TensorSpec`.")
            input_signature = (
                tf.TensorSpec((None, 10), dtype="float32", name="inputs"),
            )
        else:
            input_signature = (input_signature,)
        xla_saved_model.export_xla_saved_model(
            model,
            temp_filepath,
            input_signature=input_signature,
            platforms=["cpu"],
        )
        assert_signature(
            self,
            temp_filepath,
            "serving_default",
            {"inputs": ops.convert_to_numpy(ref_input)},
            {"output_0": ref_output},
        )

    def test_model_with_multiple_inputs(self):
        class TupleModel(models.Model):
            def __init__(self):
                super().__init__()
                self.dense = layers.Dense(4)

            def call(self, inputs):
                x, y = inputs
                return self.dense(ops.add(x, y))

        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model")
        model = TupleModel()
        ref_inputs = (
            np.random.normal(size=(3, 10)).astype("float32"),
            np.random.normal(size=(3, 10)).astype("float32"),
        )
        ref_output = model(ref_inputs)

        xla_saved_model.export_xla_saved_model(
            model,
            temp_filepath,
            input_signature=[
                (
                    layers.InputSpec(
                        shape=(None, 10), dtype="float32", name="x"
                    ),
                    layers.InputSpec(
                        shape=(None, 10), dtype="float32", name="y"
                    ),
                )
            ],
            platforms=["cpu"],
        )
        assert_signature(
            self,
            temp_filepath,
            "serving_default",
            {"x": ref_inputs[0], "y": ref_inputs[1]},
            {"output_0": ref_output},
        )

    def test_verbose(self):
        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model")
        model = get_model("sequential")
        # `verbose=None` defaults to True and prints the endpoints; both
        # settings must export successfully.
        xla_saved_model.export_xla_saved_model(
            model, temp_filepath, verbose=None, platforms=["cpu"]
        )
        xla_saved_model.export_xla_saved_model(
            model, temp_filepath, verbose=False, platforms=["cpu"]
        )
        self.assertTrue(
            os.path.isfile(os.path.join(temp_filepath, "manifest.pb"))
        )

    def test_export_xla_saved_model_errors(self):
        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model")

        # Model has not been built
        model = models.Sequential([layers.Dense(2)])
        with self.assertRaisesRegex(ValueError, "It must be built"):
            xla_saved_model.export_xla_saved_model(model, temp_filepath)

        # Subclassed model has not been called
        model = xla_saved_model_archive_test.CustomModel([layers.Dense(2)])
        model.build((2, 10))
        with self.assertRaisesRegex(ValueError, "It must be called"):
            xla_saved_model.export_xla_saved_model(model, temp_filepath)

        # Unsupported input signature element
        model = get_model("functional")
        with self.assertRaisesRegex(TypeError, "Unsupported x="):
            xla_saved_model.export_xla_saved_model(
                model, temp_filepath, input_signature=(123,)
            )

        # TF SavedModel-only kwargs are rejected
        with self.assertRaisesRegex(ValueError, "Unsupported keyword arg"):
            xla_saved_model.export_xla_saved_model(
                model, temp_filepath, jax2tf_kwargs={}
            )
        with self.assertRaisesRegex(ValueError, "Unsupported keyword arg"):
            xla_saved_model.export_xla_saved_model(
                model, temp_filepath, is_static=True
            )

    @parameterized.named_parameters(
        named_product(model_type=["sequential", "functional", "subclass"])
    )
    def test_model_export_method(self, model_type):
        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model")
        model = get_model(model_type)
        ref_input = np.random.normal(size=(3, 10)).astype("float32")
        ref_output = model(ref_input)

        model.export(temp_filepath, format="xla_saved_model", platforms=["cpu"])
        assert_signature(
            self,
            temp_filepath,
            "serving_default",
            {"inputs": ref_input},
            {"output_0": ref_output},
        )
        # Test with a different batch size
        other_input = np.random.normal(size=(6, 10)).astype("float32")
        assert_signature(
            self,
            temp_filepath,
            "serving_default",
            {"inputs": other_input},
            {"output_0": model(other_input)},
        )

    def test_model_export_method_forwards_kwargs(self):
        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model")
        model = get_model("sequential")
        model.export(
            temp_filepath,
            format="xla_saved_model",
            verbose=False,
            input_signature=[
                layers.InputSpec(shape=(None, 10), dtype="float32", name="x")
            ],
            platforms=["cpu"],
            warmup_examples=[{"x": np.zeros((1, 10), dtype="float32")}],
        )
        self.assertTrue(
            os.path.isfile(
                os.path.join(
                    temp_filepath, "assets.extra", "xla_serving_warmup_requests"
                )
            )
        )
        ref_input = np.random.normal(size=(2, 10)).astype("float32")
        assert_signature(
            self,
            temp_filepath,
            "serving_default",
            {"x": ref_input},
            {"output_0": model(ref_input)},
        )
