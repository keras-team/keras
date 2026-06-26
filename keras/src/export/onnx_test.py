"""Tests for ONNX exporting utilities."""

import os

import numpy as np
import onnxruntime
import pytest
from absl.testing import parameterized

from keras.src import backend
from keras.src import layers
from keras.src import models
from keras.src import ops
from keras.src import testing
from keras.src import tree
from keras.src.export import onnx
from keras.src.layers.input_spec import InputSpec as InputSpec
from keras.src.saving import saving_lib
from keras.src.testing.test_utils import named_product


class CustomModel(models.Model):
    def __init__(self, layer_list):
        super().__init__()
        self.layer_list = layer_list

    def call(self, input):
        output = input
        for layer in self.layer_list:
            output = layer(output)
        return output


def get_model(type="sequential", input_shape=(10,), layer_list=None):
    layer_list = layer_list or [
        layers.Dense(10, activation="relu"),
        layers.BatchNormalization(),
        layers.Dense(1, activation="sigmoid"),
    ]
    if type == "sequential":
        return models.Sequential(layer_list)
    elif type == "functional":
        input = output = tree.map_shape_structure(layers.Input, input_shape)
        for layer in layer_list:
            output = layer(output)
        return models.Model(inputs=input, outputs=output)
    elif type == "subclass":
        return CustomModel(layer_list)
    elif type == "lstm":
        # https://github.com/keras-team/keras/issues/21390
        inputs = layers.Input((4, 10))
        x = layers.Bidirectional(
            layers.LSTM(
                10,
                kernel_initializer="he_normal",
                return_sequences=True,
                kernel_regularizer=None,
            ),
            merge_mode="sum",
        )(inputs)
        outputs = layers.Bidirectional(
            layers.LSTM(
                10,
                kernel_initializer="he_normal",
                return_sequences=True,
                kernel_regularizer=None,
            ),
            merge_mode="concat",
        )(x)
        return models.Model(inputs=inputs, outputs=outputs)


@pytest.mark.skipif(
    backend.backend() not in ("tensorflow", "jax", "torch"),
    reason=(
        "`export_onnx` only currently supports the tensorflow, jax and torch "
        "backends."
    ),
)
class ExportONNXTest(testing.TestCase):
    @parameterized.named_parameters(
        named_product(
            model_type=["sequential", "functional", "subclass", "lstm"]
        )
    )
    def test_standard_model_export(self, model_type):
        if model_type == "lstm" and backend.backend() in ("jax", "tensorflow"):
            self.skipTest("Bidirectional LSTM is unsupported.")
        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model")
        model = get_model(model_type)
        batch_size = 3 if backend.backend() != "torch" else 1
        if model_type == "lstm":
            ref_input = np.random.normal(size=(batch_size, 4, 10))
        else:
            ref_input = np.random.normal(size=(batch_size, 10))
        ref_input = ref_input.astype("float32")
        ref_output = model(ref_input)

        onnx.export_onnx(model, temp_filepath)
        ort_session = onnxruntime.InferenceSession(temp_filepath)
        ort_inputs = {
            k.name: v for k, v in zip(ort_session.get_inputs(), [ref_input])
        }
        # cuDNN-fused LSTM reference vs. unrolled ONNX graph differ by ~3e-6.
        atol = 1e-5 if model_type == "lstm" else 1e-6
        rtol = 1e-5 if model_type == "lstm" else 1e-6
        self.assertAllClose(
            ort_session.run(None, ort_inputs)[0],
            ref_output,
            atol=atol,
            rtol=rtol,
            tpu_atol=1e-3,
            tpu_rtol=1e-2,
        )
        # Test with a different batch size
        ort_inputs = {
            k.name: v
            for k, v in zip(
                ort_session.get_inputs(),
                [np.concatenate([ref_input, ref_input], axis=0)],
            )
        }
        ort_session.run(None, ort_inputs)

    @parameterized.named_parameters(
        named_product(struct_type=["tuple", "array", "dict"])
    )
    def test_model_with_input_structure(self, struct_type):
        if backend.backend() == "torch" and struct_type == "tuple":
            self.skipTest("The torch backend doesn't support this structure.")

        class TupleModel(models.Model):
            def call(self, inputs):
                x, y = inputs
                return ops.add(x, y)

        class ArrayModel(models.Model):
            def call(self, inputs):
                x = inputs[0]
                y = inputs[1]
                return ops.add(x, y)

        class DictModel(models.Model):
            def call(self, inputs):
                x = inputs["x"]
                y = inputs["y"]
                return ops.add(x, y)

        batch_size = 3 if backend.backend() != "torch" else 1
        ref_input = np.random.normal(size=(batch_size, 10)).astype("float32")
        if struct_type == "tuple":
            model = TupleModel()
            ref_input = (ref_input, ref_input * 2)
        elif struct_type == "array":
            model = ArrayModel()
            ref_input = [ref_input, ref_input * 2]
        elif struct_type == "dict":
            model = DictModel()
            ref_input = {"x": ref_input, "y": ref_input * 2}

        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model")
        ref_output = model(tree.map_structure(ops.convert_to_tensor, ref_input))

        onnx.export_onnx(model, temp_filepath)
        ort_session = onnxruntime.InferenceSession(temp_filepath)
        if isinstance(ref_input, dict):
            ort_inputs = {
                k.name: v
                for k, v in zip(ort_session.get_inputs(), ref_input.values())
            }
        else:
            ort_inputs = {
                k.name: v for k, v in zip(ort_session.get_inputs(), ref_input)
            }
        self.assertAllClose(ort_session.run(None, ort_inputs)[0], ref_output)

        # Test with keras.saving_lib
        temp_filepath = os.path.join(
            self.get_temp_dir(), "exported_model.keras"
        )
        saving_lib.save_model(model, temp_filepath)
        revived_model = saving_lib.load_model(
            temp_filepath,
            {
                "TupleModel": TupleModel,
                "ArrayModel": ArrayModel,
                "DictModel": DictModel,
            },
        )
        self.assertAllClose(revived_model(ref_input), ref_output)
        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model2")
        onnx.export_onnx(revived_model, temp_filepath)

        # Test with a different batch size
        bigger_ref_input = tree.map_structure(
            lambda x: np.concatenate([x, x], axis=0), ref_input
        )
        if isinstance(bigger_ref_input, dict):
            bigger_ort_inputs = {
                k.name: v
                for k, v in zip(
                    ort_session.get_inputs(), bigger_ref_input.values()
                )
            }
        else:
            bigger_ort_inputs = {
                k.name: v
                for k, v in zip(ort_session.get_inputs(), bigger_ref_input)
            }
        ort_session.run(None, bigger_ort_inputs)

    def test_model_with_multiple_inputs(self):
        class TwoInputsModel(models.Model):
            def call(self, x, y):
                return x + y

            def build(self, y_shape, x_shape):
                self.built = True

        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model")
        model = TwoInputsModel()
        batch_size = 3 if backend.backend() != "torch" else 1
        ref_input_x = np.random.normal(size=(batch_size, 10)).astype("float32")
        ref_input_y = np.random.normal(size=(batch_size, 10)).astype("float32")
        ref_output = model(ref_input_x, ref_input_y)

        onnx.export_onnx(model, temp_filepath)
        ort_session = onnxruntime.InferenceSession(temp_filepath)
        ort_inputs = {
            k.name: v
            for k, v in zip(
                ort_session.get_inputs(), [ref_input_x, ref_input_y]
            )
        }
        self.assertAllClose(ort_session.run(None, ort_inputs)[0], ref_output)
        # Test with a different batch size
        ort_inputs = {
            k.name: v
            for k, v in zip(
                ort_session.get_inputs(),
                [
                    np.concatenate([ref_input_x, ref_input_x], axis=0),
                    np.concatenate([ref_input_y, ref_input_y], axis=0),
                ],
            )
        }
        ort_session.run(None, ort_inputs)

    @parameterized.named_parameters(named_product(opset_version=[None, 17]))
    def test_export_with_opset_version(self, opset_version):
        import onnx as onnx_lib

        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model")
        model = get_model("sequential")
        batch_size = 3 if backend.backend() != "torch" else 1
        ref_input = np.random.normal(size=(batch_size, 10))
        ref_input = ref_input.astype("float32")
        ref_output = model(ref_input)

        onnx.export_onnx(
            model, temp_filepath, opset_version=opset_version, verbose=True
        )
        ort_session = onnxruntime.InferenceSession(temp_filepath)
        ort_inputs = {
            k.name: v for k, v in zip(ort_session.get_inputs(), [ref_input])
        }
        self.assertAllClose(
            ort_session.run(None, ort_inputs)[0],
            ref_output,
            tpu_atol=1e-3,
            tpu_rtol=2e-3,
        )

        if opset_version is not None:
            onnx_model = onnx_lib.load(temp_filepath)
            self.assertEqual(onnx_model.opset_import[0].version, opset_version)

    def test_export_with_input_names(self):
        """Test ONNX export uses InputSpec.name for input names."""
        import onnx as onnx_lib

        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model")
        model = get_model("sequential")
        batch_size = 3 if backend.backend() != "torch" else 1
        ref_input = np.random.normal(size=(batch_size, 10)).astype("float32")
        ref_output = model(ref_input)

        # Test with custom input name
        input_spec = [
            InputSpec(
                name="custom_input", shape=(batch_size, 10), dtype="float32"
            )
        ]
        onnx.export_onnx(model, temp_filepath, input_signature=input_spec)

        onnx_model = onnx_lib.load(temp_filepath)
        input_names = [input.name for input in onnx_model.graph.input]
        self.assertIn("custom_input", input_names)

        ort_session = onnxruntime.InferenceSession(temp_filepath)
        ort_inputs = {
            k.name: v for k, v in zip(ort_session.get_inputs(), [ref_input])
        }
        self.assertAllClose(
            ort_session.run(None, ort_inputs)[0],
            ref_output,
            tpu_atol=1e-3,
            tpu_rtol=1e-2,
        )

    @parameterized.named_parameters(
        named_product(
            model_type=["sequential", "functional"],
            dynamic_type=["batch_only", "height_width"],
        )
    )
    def test_dynamic_shapes_export(self, model_type, dynamic_type):
        """Test ONNX export with various dynamic shape configurations.

        Tests two scenarios:
        - batch_only: Only batch dimension is dynamic, spatial dims fixed
        - height_width: Batch, height, width are dynamic, channels fixed
        """

        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model")
        channels_last = backend.image_data_format() == "channels_last"

        # Define input shapes based on dynamic type, ordered for the active
        # image data format so the channel axis is always static. The torch CI
        # runs with `channels_first`, where a dynamic spatial axis would
        # otherwise land on the channel position and break `Conv2D.build`.
        if dynamic_type == "batch_only":
            # Only batch is dynamic, spatial and channels fixed.
            if channels_last:
                input_shape = (32, 32, 3)
                test_shapes = [(1, 32, 32, 3), (2, 32, 32, 3), (4, 32, 32, 3)]
            else:
                input_shape = (3, 32, 32)
                test_shapes = [(1, 3, 32, 32), (2, 3, 32, 32), (4, 3, 32, 32)]
        elif dynamic_type == "height_width":
            # Height and width are dynamic, channels fixed.
            if channels_last:
                input_shape = (None, None, 3)
                test_shapes = [(1, 28, 28, 3), (1, 64, 64, 3), (1, 128, 96, 3)]
            else:
                input_shape = (3, None, None)
                test_shapes = [(1, 3, 28, 28), (1, 3, 64, 64), (1, 3, 128, 96)]

        # Create model with appropriate layers for dynamic shapes
        layer_list = [
            layers.Conv2D(16, 3, padding="same", activation="relu"),
            layers.GlobalAveragePooling2D(),
            layers.Dense(10, activation="softmax"),
        ]

        if model_type == "sequential":
            model = models.Sequential(
                [layers.Input(shape=input_shape)] + layer_list
            )
        elif model_type == "functional":
            input_layer = layers.Input(shape=input_shape)
            output = input_layer
            for layer in layer_list:
                output = layer(output)
            model = models.Model(inputs=input_layer, outputs=output)

        # Build model with initial input
        initial_input = np.random.normal(size=test_shapes[0]).astype(np.float32)
        model(initial_input)

        # Export to ONNX
        onnx.export_onnx(model, temp_filepath)

        # Verify with ONNX Runtime
        ort_session = onnxruntime.InferenceSession(temp_filepath)
        input_info = ort_session.get_inputs()[0]

        # Check that dynamic dimensions are preserved
        input_shape_onnx = input_info.shape
        channel_axis = len(input_shape_onnx) - 1 if channels_last else 1
        if dynamic_type == "batch_only":
            # Batch should be dynamic, others static
            self.assertTrue(isinstance(input_shape_onnx[0], str))  # Dynamic
            expected_static = [32, 32, 3] if channels_last else [3, 32, 32]
            self.assertEqual(input_shape_onnx[1:], expected_static)  # Static
        elif dynamic_type == "height_width":
            # Batch, height, width should be dynamic, channels static
            for axis, dim in enumerate(input_shape_onnx):
                if axis == channel_axis:
                    self.assertEqual(dim, 3)  # Static
                else:
                    self.assertTrue(isinstance(dim, str))  # Dynamic

        # Test inference with different input shapes
        for test_shape in test_shapes:
            test_input = np.random.randn(*test_shape).astype(np.float32)
            ort_inputs = {input_info.name: test_input}
            result = ort_session.run(None, ort_inputs)

            # Verify output shape matches expected batch size
            expected_batch_size = test_shape[0]
            self.assertEqual(result[0].shape[0], expected_batch_size)
            self.assertEqual(result[0].shape[1], 10)  # Number of classes

    def test_multi_input_dynamic_shapes(self):
        """Test ONNX export with multi-input model having dynamic shapes."""

        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model")
        channels_last = backend.image_data_format() == "channels_last"

        # Create multi-input model with dynamic shapes. The image input is
        # ordered for the active data format so the dynamic spatial axes never
        # collide with the static channel axis (torch CI runs channels_first).
        text_input = layers.Input(
            shape=(None, 64), name="text_input"
        )  # Variable sequence length
        image_input = layers.Input(
            shape=(None, None, 3) if channels_last else (3, None, None),
            name="image_input",
        )  # Variable image size

        # Process text input. The text tensor is laid out (batch, steps,
        # features) regardless of the image data format, so pin the pooling
        # to `channels_last` to pool over the dynamic steps axis.
        text_features = layers.Dense(128, activation="relu")(text_input)
        text_pooled = layers.GlobalAveragePooling1D(
            data_format="channels_last"
        )(text_features)

        # Process image input
        image_features = layers.Conv2D(32, 1, activation="relu")(
            image_input
        )  # Use 1x1 conv to avoid size issues
        image_pooled = layers.GlobalAveragePooling2D()(image_features)

        # Combine features
        combined = layers.Concatenate()([text_pooled, image_pooled])
        output = layers.Dense(5, activation="softmax")(combined)

        model = models.Model(inputs=[text_input, image_input], outputs=output)

        # Build model
        sample_text = np.random.normal(size=(1, 20, 64)).astype(np.float32)
        sample_image = np.random.normal(
            size=(1, 32, 32, 3) if channels_last else (1, 3, 32, 32)
        ).astype(np.float32)
        model([sample_text, sample_image])

        # Export to ONNX
        onnx.export_onnx(model, temp_filepath)

        # Verify with ONNX Runtime
        ort_session = onnxruntime.InferenceSession(temp_filepath)
        inputs_info = ort_session.get_inputs()

        # Check that both inputs have dynamic dimensions
        text_shape = inputs_info[0].shape
        image_shape = inputs_info[1].shape

        # Text input: [batch, seq_len, features] - batch and seq_len dynamic
        self.assertTrue(isinstance(text_shape[0], str))  # Dynamic
        self.assertTrue(isinstance(text_shape[1], str))  # Dynamic
        self.assertEqual(text_shape[2], 64)  # Static

        # Image input: batch + spatial dynamic, channels static. The channel
        # axis position depends on the active data format.
        image_channel_axis = len(image_shape) - 1 if channels_last else 1
        for axis, dim in enumerate(image_shape):
            if axis == image_channel_axis:
                self.assertEqual(dim, 3)  # Static
            else:
                self.assertTrue(isinstance(dim, str))  # Dynamic

        # Test inference with different input shapes
        def image_shape_for(batch, height, width):
            if channels_last:
                return (batch, height, width, 3)
            return (batch, 3, height, width)

        test_cases = [
            ((1, 10, 64), image_shape_for(1, 28, 28)),
            ((2, 15, 64), image_shape_for(2, 64, 64)),
            ((1, 25, 64), image_shape_for(1, 48, 32)),
        ]

        for text_shape, image_shape in test_cases:
            text_input_data = np.random.randn(*text_shape).astype(np.float32)
            image_input_data = np.random.randn(*image_shape).astype(np.float32)

            ort_inputs = {
                inputs_info[0].name: text_input_data,
                inputs_info[1].name: image_input_data,
            }
            result = ort_session.run(None, ort_inputs)

            # Verify output shape matches expected batch size
            expected_batch_size = text_shape[0]
            self.assertEqual(result[0].shape[0], expected_batch_size)
            self.assertEqual(result[0].shape[1], 5)  # Number of classes

    @pytest.mark.skipif(
        backend.backend() != "torch",
        reason="The dynamic recurrent guard is specific to the torch backend.",
    )
    def test_dynamic_sequence_recurrent_raises(self):
        """A recurrent layer with a dynamic time axis must fail loudly.

        TorchScript bakes the sequence length into the exported graph, so the
        torch backend cannot honor a dynamic time dimension for recurrent
        layers. Exporting one silently produced wrong outputs (LSTM/GRU) or a
        runtime crash (SimpleRNN), so it now raises at export time instead.
        """
        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model")
        model = models.Sequential(
            [layers.Input(shape=(None, 16)), layers.LSTM(8)]
        )
        model(np.zeros((1, 12, 16), dtype="float32"))
        with self.assertRaisesRegex(ValueError, "dynamic time dimension"):
            onnx.export_onnx(model, temp_filepath, verbose=False)

    @pytest.mark.skipif(
        backend.backend() != "torch",
        reason="The dynamic recurrent guard is specific to the torch backend.",
    )
    def test_static_sequence_recurrent_dynamic_batch_exports(self):
        """A recurrent model with a static time axis still exports.

        Only the time dimension is constrained: a dynamic batch dimension is
        fine because the sequence length stays fixed, so the guard must not
        reject this case.
        """
        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model")
        model = models.Sequential(
            [layers.Input(shape=(12, 16)), layers.LSTM(8)]
        )
        ref_input = np.random.normal(size=(2, 12, 16)).astype("float32")
        ref_output = model(ref_input)

        onnx.export_onnx(model, temp_filepath, verbose=False)
        ort_session = onnxruntime.InferenceSession(temp_filepath)
        # Batch dimension stays dynamic.
        self.assertTrue(isinstance(ort_session.get_inputs()[0].shape[0], str))
        ort_inputs = {
            k.name: v for k, v in zip(ort_session.get_inputs(), [ref_input])
        }
        self.assertAllClose(
            ort_session.run(None, ort_inputs)[0],
            ref_output,
            atol=1e-4,
        )

    @pytest.mark.skipif(
        backend.backend() != "torch",
        reason="The dynamic recurrent guard is specific to the torch backend.",
    )
    def test_static_recurrent_with_dynamic_nonrecurrent_input_exports(self):
        """A static RNN alongside a dynamic non-recurrent input still exports.

        The guard inspects each recurrent layer's own sequence shape, so a
        dynamic axis that only feeds a non-recurrent branch (here a `Conv2D`
        with dynamic spatial dims) must not trip it while the `LSTM` sequence
        length is static.
        """
        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model")
        channels_last = backend.image_data_format() == "channels_last"
        seq_input = layers.Input(shape=(10, 16), name="seq_input")
        image_input = layers.Input(
            shape=(None, None, 3) if channels_last else (3, None, None),
            name="image_input",
        )
        seq_features = layers.LSTM(8)(seq_input)
        image_features = layers.GlobalAveragePooling2D()(
            layers.Conv2D(8, 1)(image_input)
        )
        combined = layers.Concatenate()([seq_features, image_features])
        output = layers.Dense(4)(combined)
        model = models.Model(inputs=[seq_input, image_input], outputs=output)
        sample_image = (
            np.zeros((1, 8, 8, 3), dtype="float32")
            if channels_last
            else np.zeros((1, 3, 8, 8), dtype="float32")
        )
        model([np.zeros((1, 10, 16), dtype="float32"), sample_image])

        # Must not raise despite the dynamic spatial dims on the image input.
        onnx.export_onnx(model, temp_filepath, verbose=False)
        ort_session = onnxruntime.InferenceSession(temp_filepath)
        self.assertEqual(len(ort_session.get_inputs()), 2)
