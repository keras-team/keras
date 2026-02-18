"""Tests for PyTorch ExportedProgram exporting utilities."""

import os

import numpy as np
import pytest
from absl.testing import parameterized

from keras.src import backend
from keras.src import layers
from keras.src import models
from keras.src import testing
from keras.src import tree
from keras.src.testing.test_utils import named_product

# Tolerance constants for numerical comparison in torch export tests
_DEFAULT_ATOL = 1e-5
_DEFAULT_RTOL = 1e-5


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
        model = models.Sequential(layer_list)
        model.build(input_shape=(None,) + input_shape)
        return model
    if type == "functional":
        input = output = tree.map_shape_structure(layers.Input, input_shape)
        for layer in layer_list:
            output = layer(output)
        return models.Model(inputs=input, outputs=output)
    if type == "subclass":
        model = CustomModel(layer_list)
        model.build(input_shape=(None,) + input_shape)
        dummy_input = np.zeros((1,) + input_shape, dtype=np.float32)
        _ = model(dummy_input)
        return model
    if type == "multi_input":
        input1 = layers.Input(shape=input_shape, name="input1")
        input2 = layers.Input(shape=input_shape, name="input2")
        x1 = layers.Dense(10, activation="relu")(input1)
        x2 = layers.Dense(10, activation="relu")(input2)
        combined = layers.concatenate([x1, x2])
        output = layers.Dense(1, activation="sigmoid")(combined)
        return models.Model(inputs=[input1, input2], outputs=output)
    if type == "multi_output":
        inputs = layers.Input(shape=input_shape)
        shared = layers.Dense(20, activation="relu")(inputs)
        output1 = layers.Dense(1, activation="sigmoid", name="output1")(shared)
        output2 = layers.Dense(3, activation="softmax", name="output2")(shared)
        return models.Model(inputs=inputs, outputs=[output1, output2])
    raise ValueError(f"Unknown model type: {type}")


def _to_numpy(x):
    """Convert any tensor to numpy, handling MPS device tensors."""
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    if hasattr(x, "numpy"):
        return np.array(x)
    return np.array(x)


def _convert_to_numpy(structure):
    return tree.map_structure(_to_numpy, structure)


def _get_torch_device():
    """Get the device that Keras torch backend uses."""
    from keras.src.backend.torch.core import get_device

    return get_device()


def _to_torch_tensor(np_array):
    """Convert numpy array to torch tensor on the correct device."""
    import torch

    return torch.tensor(np_array).to(_get_torch_device())


@pytest.mark.skipif(
    backend.backend() != "torch",
    reason="`export_torch` only supports the PyTorch backend.",
)
class ExportTorchTest(testing.TestCase):
    """Test suite for PyTorch ExportedProgram export functionality."""

    @parameterized.named_parameters(
        named_product(model_type=["sequential", "functional"])
    )
    def test_standard_model_export(self, model_type):
        """Test exporting standard model types to PyTorch format."""
        import torch

        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model.pt2")
        model = get_model(model_type)
        ref_input = np.random.normal(size=(1, 10)).astype("float32")
        ref_output = _convert_to_numpy(model(ref_input))

        model.export(temp_filepath, format="torch")
        self.assertTrue(os.path.exists(temp_filepath))

        # Load and verify - input must be on same device as model
        loaded_program = torch.export.load(temp_filepath)
        loaded_model = loaded_program.module()
        loaded_output = loaded_model(_to_torch_tensor(ref_input))
        loaded_output = _to_numpy(loaded_output)

        self.assertAllClose(
            ref_output, loaded_output, atol=_DEFAULT_ATOL, rtol=_DEFAULT_RTOL
        )

    def test_export_subclass_model(self):
        """Test exporting subclass models."""
        import torch

        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model.pt2")
        model = get_model("subclass")
        ref_input = np.random.normal(size=(1, 10)).astype("float32")
        ref_output = _convert_to_numpy(model(ref_input))

        model.export(temp_filepath, format="torch")
        self.assertTrue(os.path.exists(temp_filepath))

        loaded_program = torch.export.load(temp_filepath)
        loaded_model = loaded_program.module()
        loaded_output = loaded_model(_to_torch_tensor(ref_input))
        loaded_output = _to_numpy(loaded_output)

        self.assertAllClose(
            ref_output, loaded_output, atol=_DEFAULT_ATOL, rtol=_DEFAULT_RTOL
        )

    def test_model_with_multiple_inputs(self):
        """Test exporting models with multiple inputs."""
        import torch

        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model.pt2")
        model = get_model("multi_input")
        ref_input1 = np.random.normal(size=(1, 10)).astype("float32")
        ref_input2 = np.random.normal(size=(1, 10)).astype("float32")
        ref_output = _convert_to_numpy(model([ref_input1, ref_input2]))

        model.export(temp_filepath, format="torch")
        self.assertTrue(os.path.exists(temp_filepath))

        loaded_program = torch.export.load(temp_filepath)
        loaded_model = loaded_program.module()
        # Multi-input models exported from Keras expect a list of inputs
        loaded_output = loaded_model(
            [_to_torch_tensor(ref_input1), _to_torch_tensor(ref_input2)]
        )
        loaded_output = _to_numpy(loaded_output)

        self.assertAllClose(
            ref_output, loaded_output, atol=_DEFAULT_ATOL, rtol=_DEFAULT_RTOL
        )

    def test_multi_output_model_export(self):
        """Test exporting multi-output models."""
        import torch

        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model.pt2")
        model = get_model("multi_output")
        ref_input = np.random.normal(size=(1, 10)).astype("float32")
        ref_outputs = _convert_to_numpy(model(ref_input))

        model.export(temp_filepath, format="torch")
        self.assertTrue(os.path.exists(temp_filepath))

        loaded_program = torch.export.load(temp_filepath)
        loaded_model = loaded_program.module()
        loaded_outputs = loaded_model(_to_torch_tensor(ref_input))

        # Multi-output may return a tuple or list
        if isinstance(loaded_outputs, (tuple, list)):
            for ref_out, loaded_out in zip(ref_outputs, loaded_outputs):
                loaded_np = _to_numpy(loaded_out)
                self.assertAllClose(
                    ref_out, loaded_np, atol=_DEFAULT_ATOL, rtol=_DEFAULT_RTOL
                )
        else:
            loaded_np = _to_numpy(loaded_outputs)
            self.assertAllClose(
                ref_outputs, loaded_np, atol=_DEFAULT_ATOL, rtol=_DEFAULT_RTOL
            )

    def test_export_with_custom_input_signature(self):
        """Test exporting with custom input signature specification."""
        import torch

        model = get_model("sequential")
        ref_input = np.random.normal(size=(1, 10)).astype("float32")
        model(ref_input)  # Build

        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model.pt2")
        input_signature = [layers.InputSpec(shape=(1, 10), dtype="float32")]

        model.export(
            temp_filepath,
            format="torch",
            input_signature=input_signature,
        )
        self.assertTrue(os.path.exists(temp_filepath))

        # Verify we can load and run
        loaded_program = torch.export.load(temp_filepath)
        loaded_model = loaded_program.module()
        test_input = torch.randn(1, 10).to(_get_torch_device())
        output = loaded_model(test_input)
        self.assertEqual(_to_numpy(output).shape[-1], 1)

        # Numeric verification
        ref_output = _convert_to_numpy(model(_to_numpy(test_input)))
        self.assertAllClose(
            ref_output,
            _to_numpy(output),
            atol=_DEFAULT_ATOL,
            rtol=_DEFAULT_RTOL,
        )

    def test_export_with_verbose(self):
        """Test export with verbose output."""
        model = get_model("sequential")
        ref_input = np.random.normal(size=(1, 10)).astype("float32")
        model(ref_input)

        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model.pt2")

        # Export with verbose=True should not raise
        model.export(temp_filepath, format="torch", verbose=True)
        self.assertTrue(os.path.exists(temp_filepath))

        # Numeric verification
        import torch

        loaded_program = torch.export.load(temp_filepath)
        loaded_model = loaded_program.module()

        ref_output = _convert_to_numpy(model(ref_input))
        loaded_output = loaded_model(_to_torch_tensor(ref_input))

        self.assertAllClose(
            ref_output,
            _to_numpy(loaded_output),
            atol=_DEFAULT_ATOL,
            rtol=_DEFAULT_RTOL,
        )

    def test_export_error_handling(self):
        """Test error handling in export API."""
        model = get_model("sequential")
        ref_input = np.random.normal(size=(1, 10)).astype("float32")
        model(ref_input)

        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model.pt2")

        # Test with invalid format
        with self.assertRaises(ValueError):
            model.export(temp_filepath, format="invalid_format")

    def test_export_invalid_filepath(self):
        """Test that export fails with invalid file extension."""
        model = get_model("sequential")
        ref_input = np.random.normal(size=(1, 10)).astype("float32")
        model(ref_input)

        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model.txt")

        # Should raise ValueError for wrong extension
        with self.assertRaises(ValueError):
            model.export(temp_filepath, format="torch")

    def test_model_with_input_structure(self):
        """Test exporting models with structured inputs (tuple/array/dict)."""
        import torch

        # Define basic input
        ref_input_arr = np.random.normal(size=(1, 10)).astype("float32")

        # Case 1: Tuple input
        input1 = layers.Input(shape=(10,), name="input_1")
        input2 = layers.Input(shape=(10,), name="input_2")
        output = layers.Add()([input1, input2])
        model_tuple = models.Model(inputs=[input1, input2], outputs=output)
        ref_input_tuple = (ref_input_arr, ref_input_arr * 2)

        temp_filepath = os.path.join(self.get_temp_dir(), "tuple_model.pt2")
        model_tuple.export(temp_filepath, format="torch")

        loaded_program = torch.export.load(temp_filepath)
        loaded_model = loaded_program.module()
        # Exported model expects a tuple argument or expanded args.
        # Keras models with list inputs expects list of args or single
        # list/tuple. `get_input_signature` returns a list of specs.
        # `model.export` converts this to a tuple of tensors: `(t1, t2)`.
        # The traced model expects `(t1, t2)`.
        # But `model.export` logic: sample_inputs = (t1, t2)
        # model(*sample_inputs) -> model(t1, t2)
        # So the model actually expects *args.
        # For multi-input functional models, `get_input_signature`
        # returns a list containing a list of specs.
        # `model.export` converts this to `([t1, t2],)`.
        # So the model expects a single argument which is a list/tuple.
        loaded_output = loaded_model(
            [
                _to_torch_tensor(ref_input_tuple[0]),
                _to_torch_tensor(ref_input_tuple[1]),
            ]
        )
        self.assertAllClose(
            _convert_to_numpy(model_tuple(ref_input_tuple)),
            _to_numpy(loaded_output),
            atol=_DEFAULT_ATOL,
            rtol=_DEFAULT_RTOL,
        )

        # Case 2: List input (Array)
        # Same model, but input passed as list
        temp_filepath = os.path.join(self.get_temp_dir(), "list_model.pt2")
        model_tuple.export(
            temp_filepath, format="torch"
        )  # Re-export same model is fine

        loaded_program = torch.export.load(temp_filepath)
        loaded_model = loaded_program.module()
        # Same signature
        loaded_output = loaded_model(
            [
                _to_torch_tensor(ref_input_arr),
                _to_torch_tensor(ref_input_arr * 2),
            ]
        )
        self.assertAllClose(
            _convert_to_numpy(model_tuple([ref_input_arr, ref_input_arr * 2])),
            _to_numpy(loaded_output),
            atol=_DEFAULT_ATOL,
            rtol=_DEFAULT_RTOL,
        )

        # Case 3: Dict input
        input_x = layers.Input(shape=(10,), name="x")
        input_y = layers.Input(shape=(10,), name="y")
        output = layers.Add()([input_x, input_y])
        model_dict = models.Model(
            inputs={"x": input_x, "y": input_y}, outputs=output
        )
        ref_input_dict = {"x": ref_input_arr, "y": ref_input_arr * 2}

        temp_filepath = os.path.join(self.get_temp_dir(), "dict_model.pt2")
        model_dict.export(temp_filepath, format="torch")

        loaded_program = torch.export.load(temp_filepath)
        loaded_model = loaded_program.module()
        # For dict inputs, the input signature is a dict spec.
        # sample_inputs = ({'x': t1, 'y': t2},) containing one
        # element which is the dict. So the model expects a single
        # dict argument.
        loaded_output = loaded_model(
            {
                "x": _to_torch_tensor(ref_input_dict["x"]),
                "y": _to_torch_tensor(ref_input_dict["y"]),
            }
        )
        self.assertAllClose(
            _convert_to_numpy(model_dict(ref_input_dict)),
            _to_numpy(loaded_output),
            atol=_DEFAULT_ATOL,
            rtol=_DEFAULT_RTOL,
        )

    def test_model_with_named_inputs(self):
        """Test that exported models preserve input names in signature."""
        import torch

        input_x = layers.Input(shape=(10,), name="input_x")
        input_y = layers.Input(shape=(10,), name="input_y")
        output = layers.Add()([input_x, input_y])
        model = models.Model(
            inputs={"x": input_x, "y": input_y}, outputs=output
        )

        temp_filepath = os.path.join(self.get_temp_dir(), "named_inputs.pt2")
        model.export(temp_filepath, format="torch")

        loaded_program = torch.export.load(temp_filepath)
        signature = loaded_program.graph_signature

        # Check that input specs contain the original names
        input_names = [spec.arg.name for spec in signature.input_specs]
        # Torch export might prefix them (e.g., args_0_x), but the
        # key part should be there
        self.assertTrue(
            any("input_x" in name or "x" in name for name in input_names)
        )
        self.assertTrue(
            any("input_y" in name or "y" in name for name in input_names)
        )

        # Verify inference
        ref_input = {
            "x": np.random.normal(size=(1, 10)).astype("float32"),
            "y": np.random.normal(size=(1, 10)).astype("float32"),
        }
        ref_output = _convert_to_numpy(model(ref_input))
        loaded_model = loaded_program.module()
        loaded_output = loaded_model(
            {
                "x": _to_torch_tensor(ref_input["x"]),
                "y": _to_torch_tensor(ref_input["y"]),
            }
        )
        self.assertAllClose(
            ref_output,
            _to_numpy(loaded_output),
            atol=_DEFAULT_ATOL,
            rtol=_DEFAULT_RTOL,
        )

    # ------------------------------------------------------------------ #
    #  Additional robustness and edge case tests
    # ------------------------------------------------------------------ #
    def test_export_with_batch_normalization(self):
        """Test export with batch normalization layer."""
        import torch

        model = models.Sequential(
            [
                layers.Dense(16, input_shape=(10,)),
                layers.BatchNormalization(),
                layers.Activation("relu"),
                layers.Dense(8),
                layers.BatchNormalization(),
                layers.Dense(1),
            ]
        )

        temp_filepath = os.path.join(self.get_temp_dir(), "bn_model.pt2")
        model.export(temp_filepath, format="torch")
        self.assertTrue(os.path.exists(temp_filepath))

        # Verify inference in eval mode
        ref_input = np.random.normal(size=(1, 10)).astype("float32")
        ref_output = _convert_to_numpy(model(ref_input, training=False))

        loaded_program = torch.export.load(temp_filepath)
        loaded_output = loaded_program.module()(_to_torch_tensor(ref_input))
        self.assertAllClose(
            ref_output, _to_numpy(loaded_output), atol=_DEFAULT_ATOL
        )

    def test_export_conv_model(self):
        """Test export with convolutional layers."""
        import torch

        model = models.Sequential(
            [
                layers.Conv2D(
                    32,
                    3,
                    activation="relu",
                    input_shape=(32, 32, 3),
                    data_format="channels_last",
                ),
                layers.MaxPooling2D(2),
                layers.Conv2D(64, 3, activation="relu"),
                layers.GlobalAveragePooling2D(),
                layers.Dense(10, activation="softmax"),
            ]
        )

        # Explicitly build the model to ensure all layer shapes are computed
        model.build((None, 32, 32, 3))

        # Test with reference input to verify model works before export
        ref_input = np.random.normal(size=(1, 32, 32, 3)).astype("float32")
        ref_output = _convert_to_numpy(model(ref_input, training=False))

        # Verify output shape is valid
        self.assertGreater(
            ref_output.size, 0, "Model output should be non-empty"
        )

        temp_filepath = os.path.join(self.get_temp_dir(), "conv_model.pt2")
        model.export(temp_filepath, format="torch")
        self.assertTrue(os.path.exists(temp_filepath))

        loaded_program = torch.export.load(temp_filepath)
        loaded_output = loaded_program.module()(_to_torch_tensor(ref_input))
        loaded_output_np = _to_numpy(loaded_output)

        # Verify loaded output shape matches reference
        msg = (
            f"Output shape mismatch: ref={ref_output.shape}, "
            f"loaded={loaded_output_np.shape}"
        )
        self.assertEqual(ref_output.shape, loaded_output_np.shape, msg)

        self.assertAllClose(ref_output, loaded_output_np, atol=_DEFAULT_ATOL)

    def test_export_functional_with_residual(self):
        """Test export functional model with residual connections."""
        import torch

        inputs = layers.Input(shape=(10,))
        x = layers.Dense(16, activation="relu")(inputs)
        residual = layers.Dense(16)(inputs)
        x = layers.Add()([x, residual])
        x = layers.Activation("relu")(x)
        outputs = layers.Dense(1)(x)
        model = models.Model(inputs=inputs, outputs=outputs)

        temp_filepath = os.path.join(self.get_temp_dir(), "residual_model.pt2")
        model.export(temp_filepath, format="torch")
        self.assertTrue(os.path.exists(temp_filepath))

        ref_input = np.random.normal(size=(1, 10)).astype("float32")
        ref_output = _convert_to_numpy(model(ref_input))

        loaded_program = torch.export.load(temp_filepath)
        loaded_output = loaded_program.module()(_to_torch_tensor(ref_input))
        self.assertAllClose(
            ref_output, _to_numpy(loaded_output), atol=_DEFAULT_ATOL
        )

    def test_export_with_concrete_shapes(self):
        """Test that exported model has concrete (non-dynamic) shapes."""
        import torch

        model = models.Sequential([layers.Dense(1, input_shape=(5,))])
        temp_filepath = os.path.join(self.get_temp_dir(), "concrete_shapes.pt2")

        # Provide concrete input signature
        from keras.src.layers.input_spec import InputSpec

        input_sig = [InputSpec(shape=(1, 5), dtype="float32")]

        model.export(temp_filepath, format="torch", input_signature=input_sig)

        loaded_program = torch.export.load(temp_filepath)
        # Verify the program was exported with static shapes
        self.assertIsNotNone(loaded_program.graph_signature)

    def test_export_with_none_in_signature(self):
        """Test export handles None batch dimension correctly."""

        model = models.Sequential([layers.Dense(1, input_shape=(5,))])
        temp_filepath = os.path.join(self.get_temp_dir(), "none_batch.pt2")

        # torch.export will replace None with a concrete value (1 by default)
        from keras.src.layers.input_spec import InputSpec

        input_sig = [InputSpec(shape=(None, 5), dtype="float32")]

        model.export(temp_filepath, format="torch", input_signature=input_sig)
        self.assertTrue(os.path.exists(temp_filepath))
