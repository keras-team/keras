import contextlib
import functools
import inspect
import io
import tempfile

from keras.src import backend
from keras.src import tree
from keras.src.export.export_utils import convert_spec_to_tensor
from keras.src.export.export_utils import get_input_signature
from keras.src.utils import io_utils
from keras.src.utils.module_utils import tensorflow as tf


def export_litert(
    model,
    filepath,
    input_signature=None,
    verbose=None,
    **kwargs,
):
    """Export the model as a LiteRT artifact for inference.

    Args:
        model: The Keras model to export.
        filepath: The path to save the exported artifact.
        input_signature: Optional input signature specification. If
            `None`, it will be inferred.
        **kwargs: Additional keyword arguments passed to the exporter.
    """
    filepath = str(filepath)
    actual_verbose = verbose if verbose is not None else True

    if not filepath.endswith(".tflite"):
        raise ValueError(
            "The LiteRT export requires the filepath to end with "
            f"'.tflite'. Got: {filepath}"
        )

    if backend.backend() == "torch":
        return export_litert_via_torch(
            model,
            filepath,
            input_signature=input_signature,
            verbose=actual_verbose,
            **kwargs,
        )

    if backend.backend() != "tensorflow":
        raise ValueError(
            "The LiteRT export API is currently only available "
            "with the TensorFlow and PyTorch backends."
        )

    exporter = LiteRTExporter(
        model=model,
        input_signature=input_signature,
        **kwargs,
    )
    exporter.export(filepath)
    if actual_verbose:
        io_utils.print_msg(f"Saved artifact at '{filepath}'.")


class LiteRTExporter:
    """Exporter for the LiteRT (TFLite) format.

    This class handles the conversion of Keras models for LiteRT runtime and
    generates a `.tflite` model file. For efficient inference on mobile and
    embedded devices, it creates a single callable signature based on the
    model's `call()` method.
    """

    def __init__(
        self,
        model,
        input_signature=None,
        **kwargs,
    ):
        """Initialize the LiteRT exporter.

        Args:
            model: The Keras model to export
            input_signature: Input signature specification (e.g., TensorFlow
                TensorSpec or list of TensorSpec)
            **kwargs: Additional export parameters
        """
        self.model = model
        self.input_signature = input_signature
        self.kwargs = kwargs

    def export(self, filepath):
        """Exports the Keras model to a TFLite file.

        Args:
            filepath: Output path for the exported model

        Returns:
            Path to exported model
        """
        if self.input_signature is None:
            # Use the standard get_input_signature which handles all model
            # types and preserves nested structures (dicts, lists, etc.)
            self.input_signature = get_input_signature(self.model)

        # Normalize input_signature for tf.function: must be a list/tuple
        # of specs, one per positional argument. Wrap bare structures as
        # a single argument.
        if not isinstance(self.input_signature, (list, tuple)):
            self.input_signature = [self.input_signature]

        if not filepath.endswith(".tflite"):
            raise ValueError(
                f"The LiteRT export requires the filepath to end with "
                f"'.tflite'. Got: {filepath}"
            )

        tflite_model = self._convert_to_tflite(self.input_signature)

        with open(filepath, "wb") as f:
            f.write(tflite_model)

        return filepath

    def _convert_to_tflite(self, input_signature):
        """Converts the Keras model to TFLite format.

        Uses Keras ExportArchive as an intermediate SavedModel step.
        This aligns with TensorFlow's official Keras 3 TFLite conversion path:
        ExportArchive -> SavedModel -> from_saved_model.

        Args:
            input_signature: Input signature for the model to convert.

        Returns:
            A bytes object containing the serialized TFLite model.
        """
        from keras.src import export as keras_export

        with tempfile.TemporaryDirectory() as saved_model_dir:
            archive = keras_export.ExportArchive()
            archive.track(self.model)
            archive.add_endpoint(
                "serve",
                functools.partial(self.model.__call__, training=False),
                input_signature=input_signature,
            )
            archive.write_out(saved_model_dir, verbose=False)

            converter = tf.lite.TFLiteConverter.from_saved_model(
                saved_model_dir
            )
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,
                tf.lite.OpsSet.SELECT_TF_OPS,
            ]
            converter.experimental_enable_resource_variables = True
            self._apply_converter_kwargs(converter)
            return converter.convert()

    def _apply_converter_kwargs(self, converter):
        """Apply additional converter settings from kwargs.

        Args:
            converter: tf.lite.TFLiteConverter instance to configure

        Raises:
            ValueError: If any kwarg is not a valid converter attribute
        """
        for attr, value in self.kwargs.items():
            if attr == "target_spec" and isinstance(value, dict):
                # Handle nested target_spec settings
                for spec_key, spec_value in value.items():
                    if hasattr(converter.target_spec, spec_key):
                        setattr(converter.target_spec, spec_key, spec_value)
                    else:
                        raise ValueError(
                            f"Unknown target_spec attribute '{spec_key}'"
                        )
            elif hasattr(converter, attr):
                setattr(converter, attr, value)
            else:
                raise ValueError(f"Unknown converter attribute '{attr}'")


def export_litert_via_torch(
    model, filepath, input_signature=None, verbose=None, **kwargs
):
    """Export Keras model to LiteRT via PyTorch backend."""
    from keras.src.backend.torch.core import device_scope
    from keras.src.backend.torch.core import get_device

    device = get_device()

    # litert_torch internally imports JAX and unconditionally enables
    # jax_enable_x64, which breaks dtype-sensitive tests elsewhere.
    # We preserve the original setting and restore it after conversion.
    with _preserve_jax_x64_state():
        try:
            import litert_torch
        except ImportError:
            raise ImportError(
                "To export to LiteRT with the PyTorch backend, "
                "you must install the `litert-torch` package. "
                "Install via: pip install litert-torch"
            )

        if device != "cpu":
            for v in model.variables:
                v.value.data = v.value.data.to("cpu")

        try:
            with device_scope("cpu"):
                if input_signature is None:
                    input_signature = get_input_signature(model)

                sample_inputs = tree.map_structure(
                    lambda x: convert_spec_to_tensor(x, replace_none_number=1),
                    input_signature,
                )
                sample_inputs = tuple(sample_inputs)

                # Although Keras models manage training mode via the
                # `training` argument to `call()`, `litert_torch.convert`
                # checks `module.training` and warns when it is `True`.
                # Calling `eval()` silences that warning and ensures any
                # PyTorch-native submodules are in inference mode.
                model.eval()

                litert_torch_kwargs = _prepare_litert_kwargs(
                    kwargs, litert_torch
                )

                with _silence_output(verbose is False):
                    try:
                        edge_model = litert_torch.convert(
                            model, sample_inputs, **litert_torch_kwargs
                        )
                    except Exception as e:
                        raise RuntimeError(
                            "Failed to convert PyTorch model to LiteRT. "
                            "Common causes: unsupported operations, "
                            "dynamic shapes, or complex control flow. "
                            f"Original error: {e}"
                        ) from e

                    edge_model.export(filepath)
        finally:
            if device != "cpu":
                for v in model.variables:
                    v.value.data = v.value.data.to(device)

    if verbose is not False:
        io_utils.print_msg(f"Saved artifact at '{filepath}'.")

    return filepath


def _prepare_litert_kwargs(kwargs, litert_torch):
    """Prepare litert_torch conversion kwargs from user-provided arguments."""
    litert_torch_kwargs = {}

    valid_litert_torch_args = _get_litert_torch_kwarg_names(litert_torch)
    for k, v in kwargs.items():
        if k in valid_litert_torch_args:
            litert_torch_kwargs[k] = v

    if "optimizations" in kwargs and "quant_config" not in litert_torch_kwargs:
        quant_cfg = _create_quant_config_from_optimizations(
            kwargs["optimizations"], litert_torch
        )
        if quant_cfg is not None:
            litert_torch_kwargs["quant_config"] = quant_cfg

    unsupported_args = sorted(
        key
        for key in kwargs
        if key not in valid_litert_torch_args and key != "optimizations"
    )
    if unsupported_args:
        supported_args = sorted(valid_litert_torch_args | {"optimizations"})
        raise ValueError(
            "Unsupported arguments for LiteRT export with the PyTorch "
            f"backend: {', '.join(unsupported_args)}. Supported arguments "
            f"are: {', '.join(supported_args)}."
        )

    return litert_torch_kwargs


def _get_litert_torch_kwarg_names(litert_torch):
    return {
        name
        for name in inspect.signature(litert_torch.convert).parameters
        if name not in {"module", "sample_args", "sample_kwargs"}
    }


@contextlib.contextmanager
def _preserve_jax_x64_state():
    try:
        import jax
    except ImportError:
        jax = None
        original_x64 = None
    else:
        original_x64 = jax.config.jax_enable_x64

    try:
        yield
    finally:
        if jax is not None:
            jax.config.update("jax_enable_x64", original_x64)


@contextlib.contextmanager
def _silence_output(should_silence):
    if not should_silence:
        yield
        return

    with contextlib.redirect_stdout(io.StringIO()):
        with contextlib.redirect_stderr(io.StringIO()):
            yield


def _create_quant_config_from_optimizations(optimizations, litert_torch):
    """Translate TFLite optimizations to litert_torch QuantConfig."""
    if not optimizations:
        return None

    try:
        from litert_torch.quantize.pt2e_quantizer import PT2EQuantizer
        from litert_torch.quantize.pt2e_quantizer import (
            get_symmetric_quantization_config,
        )
        from litert_torch.quantize.quant_config import QuantConfig
    except ImportError:
        io_utils.print_msg(
            "Warning: litert_torch quantization modules not available. "
            "Skipping quantization."
        )
        return None

    try:
        optimize_default = tf.lite.Optimize.DEFAULT
        optimize_size = getattr(tf.lite.Optimize, "OPTIMIZE_FOR_SIZE", None)
        optimize_latency = getattr(
            tf.lite.Optimize, "OPTIMIZE_FOR_LATENCY", None
        )
    except (ImportError, AttributeError):
        return None

    has_default = optimize_default in optimizations
    has_size = optimize_size and optimize_size in optimizations
    has_latency = optimize_latency and optimize_latency in optimizations

    if has_default or has_size or has_latency:
        is_dynamic = has_default and not (has_size or has_latency)
        is_per_channel = has_latency or has_size

        quant_config_obj = get_symmetric_quantization_config(
            is_per_channel=is_per_channel,
            is_dynamic=is_dynamic,
            is_qat=False,
        )

        quantizer = PT2EQuantizer()
        quantizer.set_global(quant_config_obj)

        return QuantConfig(pt2e_quantizer=quantizer)

    return None
