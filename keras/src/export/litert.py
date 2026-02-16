from keras.src import backend
from keras.src import layers
from keras.src import models
from keras.src import tree
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
    if backend.backend() == "torch":
        return export_litert_via_torch(
            model,
            filepath,
            input_signature=input_signature,
            verbose=verbose,
            **kwargs,
        )

    if backend.backend() != "tensorflow":
        raise ImportError(
            "The LiteRT export API is currently only available "
            "with the TensorFlow and PyTorch backends."
        )

    exporter = LiteRTExporter(
        model=model,
        input_signature=input_signature,
        **kwargs,
    )
    exporter.export(filepath)
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
        # 1. Resolve / infer input signature
        if self.input_signature is None:
            # Use the standard get_input_signature which handles all model types
            # and preserves nested structures (dicts, lists, etc.)
            self.input_signature = get_input_signature(self.model)

        # 2. Determine input structure and create adapter if needed
        # There are 3 cases:
        # Case 1: Single input (not nested)
        # Case 2: Flat list of inputs (list where flattened == original)
        # Case 3: Nested structure (dicts, nested lists, etc.)

        # Special handling for Functional models: get_input_signature wraps
        # the structure in a list, so unwrap it for analysis
        input_struct = self.input_signature
        if (
            isinstance(self.input_signature, list)
            and len(self.input_signature) == 1
        ):
            input_struct = self.input_signature[0]

        if not tree.is_nested(input_struct):
            # Case 1: Single input - use as-is
            model_to_convert = self.model
            signature_for_conversion = self.input_signature
        elif isinstance(input_struct, list) and len(input_struct) == len(
            tree.flatten(input_struct)
        ):
            # Case 2: Flat list of inputs - use as-is
            model_to_convert = self.model
            signature_for_conversion = self.input_signature
        else:
            # Case 3: Nested structure (dict, nested lists, etc.)
            # Create adapter model that converts flat list to nested structure
            adapted_model = self._create_nested_inputs_adapter(input_struct)

            # Flatten signature for TFLite conversion
            signature_for_conversion = tree.flatten(input_struct)

            # Use adapted model and flat list signature for conversion
            model_to_convert = adapted_model

        # Store original model reference for later use
        original_model = self.model

        # Temporarily replace self.model with the model to convert
        self.model = model_to_convert

        try:
            # Convert the model to TFLite.
            tflite_model = self._convert_to_tflite(signature_for_conversion)
        finally:
            # Restore original model
            self.model = original_model

        # Save the TFLite model to the specified file path.
        if not filepath.endswith(".tflite"):
            raise ValueError(
                f"The LiteRT export requires the filepath to end with "
                f"'.tflite'. Got: {filepath}"
            )

        with open(filepath, "wb") as f:
            f.write(tflite_model)

        return filepath

    def _create_nested_inputs_adapter(self, input_signature_struct):
        """Create an adapter model that converts flat list inputs to nested
        structure.

        This adapter allows models expecting nested inputs (dicts, lists, etc.)
        to be exported to TFLite format (which only supports positional/list
        inputs).

        Args:
            input_signature_struct: Nested structure of InputSpecs (dict, list,
                etc.)

        Returns:
            A Functional model that accepts flat list inputs and converts to
            nested
        """
        # Get flat paths to preserve names and print input mapping
        paths_and_specs = tree.flatten_with_path(input_signature_struct)
        paths = [".".join(str(e) for e in p) for p, v in paths_and_specs]
        io_utils.print_msg(f"Creating adapter for inputs: {paths}")

        # Create Input layers for TFLite (flat list-based)
        input_layers = []
        for path, spec in paths_and_specs:
            # Extract the input name from spec or path
            name = (
                spec.name
                if hasattr(spec, "name") and spec.name
                else (str(path[-1]) if path else "input")
            )

            input_layer = layers.Input(
                shape=spec.shape[1:],  # Remove batch dimension
                dtype=spec.dtype,
                name=name,
            )
            input_layers.append(input_layer)

        # Reconstruct the nested structure from flat list
        inputs_structure = tree.pack_sequence_as(
            input_signature_struct, input_layers
        )

        # Call the original model with nested inputs
        outputs = self.model(inputs_structure)

        # Build as Functional model (flat list inputs -> nested -> model ->
        # output)
        adapted_model = models.Model(inputs=input_layers, outputs=outputs)

        # Preserve the original model's variables
        adapted_model._variables = self.model.variables
        adapted_model._trainable_variables = self.model.trainable_variables
        adapted_model._non_trainable_variables = (
            self.model.non_trainable_variables
        )

        return adapted_model

    def _convert_to_tflite(self, input_signature):
        """Converts the Keras model to TFLite format.

        Returns:
            A bytes object containing the serialized TFLite model.
        """
        # Try direct conversion first for all models
        try:
            converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,
                tf.lite.OpsSet.SELECT_TF_OPS,
            ]
            # Keras 3 only supports resource variables
            converter.experimental_enable_resource_variables = True

            # Apply any additional converter settings from kwargs
            self._apply_converter_kwargs(converter)

            tflite_model = converter.convert()

            return tflite_model

        except Exception as e:
            # If direct conversion fails, raise the error with helpful message
            raise RuntimeError(
                f"Direct TFLite conversion failed. This may be due to model "
                f"complexity or unsupported operations. Error: {e}"
            ) from e

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
                # Set any valid converter attribute (optimizations, 
                # representative_dataset, etc.)
                setattr(converter, attr, value)
            else:
                raise ValueError(f"Unknown converter attribute '{attr}'")


def export_litert_via_torch(
    model, filepath, input_signature=None, verbose=None, **kwargs
):
    try:
        import litert_torch
        import torch
    except ImportError:
        raise ImportError(
            "To export to LiteRT with the PyTorch backend, "
            "you must install the `litert-torch` package. "
            "You can install it via `pip install litert-torch`."
        )

    from keras.src.export.export_utils import convert_spec_to_tensor

    # 1. Move model state to CPU to avoid MPS/CUDA-specific ops.
    # This covers three categories of tensors:
    #   a) Keras Variable instances (model.variables)
    #   b) nn.Module parameters & buffers (model.named_parameters/buffers)
    #   c) Raw tensor attributes on layers (e.g. Normalization.mean)
    original_devices = {}
    _move_model_to_cpu(model, original_devices, torch)

    # Use a CPU device scope so Keras ops (convert_to_tensor, etc.)
    # don't move intermediate tensors back to MPS/CUDA during tracing.
    from keras.src.backend.torch.core import device_scope

    with device_scope("cpu"):
        # 2. Register decompositions and fix StableHLO version
        # compatibility.  On Apple Silicon, torch.export captures
        # aten._scaled_dot_product_attention_math_for_mps which
        # litert_torch cannot lower. We register the standard
        # (non-MPS) decomposition so it decomposes into portable ops.
        _register_litert_decompositions(torch, litert_torch)

        # Force VHLO serialization to target a StableHLO version
        # compatible with the TFLite converter. Without this, JAX's
        # StableHLO (v1.13.0+) emits v2 VHLO ops (tanh_v2, etc.)
        # that TF's converter (up to ~v1.9.1) cannot parse.
        _patch_vhlo_target_version()

        # 3. Build sample inputs for conversion (always on CPU).
        if input_signature is None:
            input_signature = get_input_signature(model)

        sample_inputs = tree.map_structure(
            lambda x: convert_spec_to_tensor(x, replace_none_number=1),
            input_signature,
        )
        sample_inputs = tree.map_structure(
            lambda t: t.cpu() if hasattr(t, "cpu") else t,
            sample_inputs,
        )
        sample_inputs = tuple(sample_inputs)

        # 4. Put model in eval mode for better export quality.
        if hasattr(model, "eval"):
            model.eval()

        # 5. Convert directly via litert_torch.
        # Filter kwargs to only those supported by litert_torch.convert.
        # TFLite-specific options (optimizations, representative_dataset, etc.)
        # are not supported by litert_torch.
        litert_torch_kwargs = {}
        valid_litert_torch_args = {
            "strict_export",
            "quant_config",
            "dynamic_shapes",
            "_ai_edge_converter_flags",
            "_saved_model_dir",
        }
        for k, v in kwargs.items():
            if k in valid_litert_torch_args:
                litert_torch_kwargs[k] = v
            # Silently ignore TFLite-specific kwargs that don't apply to
            # litert_torch (e.g., optimizations, representative_dataset)

        try:
            try:
                edge_model = litert_torch.convert(
                    model, sample_inputs, **litert_torch_kwargs
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to convert PyTorch model to LiteRT: {e}"
                ) from e

            # 6. Save the .tflite model.
            edge_model.export(filepath)
        finally:
            # 7. Always restore model variables to original devices,
            # even if conversion or export failed.
            _restore_model_devices(model, original_devices, torch)

    if verbose:
        io_utils.print_msg(f"Saved LiteRT model to {filepath}")

    return filepath


def _move_model_to_cpu(model, original_devices, torch):
    """Move all model tensors to CPU for portable export.

    Handles three categories that Keras models can store tensors in:
    1. Keras ``Variable`` instances (``model.variables``).
    2. ``nn.Module`` parameters and buffers registered via PyTorch.
    3. Raw tensor attributes on Keras layers (e.g.
       ``Normalization.mean`` / ``Normalization.variance``).
    """
    # (a) Keras Variables
    try:
        for v in model.variables:
            if hasattr(v, "value") and hasattr(v.value, "data"):
                dev = str(v.value.device)
                if dev != "cpu":
                    original_devices[("var", v.path)] = dev
                    v.value.data = v.value.data.to("cpu")
    except (AttributeError, TypeError):
        # model.variables may not exist, or some variables may lack device
        # info. Continue gracefully - device-specific ops will be caught by
        # litert_torch with clearer error messages.
        pass

    # (b) nn.Module parameters and buffers
    try:
        for name, p in model.named_parameters():
            if p.device.type != "cpu":
                original_devices[("param", name)] = str(p.device)
                p.data = p.data.to("cpu")
        for name, b in model.named_buffers():
            if b.device.type != "cpu":
                original_devices[("buffer", name)] = str(b.device)
                b.data = b.data.to("cpu")
    except (AttributeError, RuntimeError):
        # Some models may not have parameters/buffers or device
        # operations may fail. Continue gracefully.
        pass

    # (c) Raw tensor attributes on layers (not Variables or registered
    #     parameters). Some preprocessing layers (e.g. Normalization)
    #     store mean/variance as plain tensor attributes.
    try:
        for layer in model._flatten_layers():
            for attr_name in list(vars(layer)):
                obj = getattr(layer, attr_name, None)
                if (
                    isinstance(obj, torch.Tensor)
                    and not isinstance(obj, torch.nn.Parameter)
                    and obj.device.type != "cpu"
                ):
                    key = ("attr", f"{layer.name}.{attr_name}")
                    original_devices[key] = str(obj.device)
                    setattr(layer, attr_name, obj.to("cpu"))
    except (AttributeError, RuntimeError):
        # Some models may not have _flatten_layers or layer attributes
        # may not support device movement. Continue gracefully.
        pass


def _restore_model_devices(model, original_devices, torch):
    """Restore model tensors to their original devices after export."""
    if not original_devices:
        return

    try:
        for v in model.variables:
            key = ("var", v.path)
            if key in original_devices:
                v.value.data = v.value.data.to(original_devices[key])
    except (AttributeError, TypeError):
        # Variables may not exist. Continue gracefully.
        pass

    try:
        for name, p in model.named_parameters():
            key = ("param", name)
            if key in original_devices:
                p.data = p.data.to(original_devices[key])
        for name, b in model.named_buffers():
            key = ("buffer", name)
            if key in original_devices:
                b.data = b.data.to(original_devices[key])
    except (AttributeError, RuntimeError):
        # Parameters/buffers may not exist or device operations may fail.
        pass

    try:
        for layer in model._flatten_layers():
            for attr_name in list(vars(layer)):
                key = ("attr", f"{layer.name}.{attr_name}")
                if key in original_devices:
                    obj = getattr(layer, attr_name, None)
                    if isinstance(obj, torch.Tensor):
                        setattr(
                            layer,
                            attr_name,
                            obj.to(original_devices[key]),
                        )
    except (AttributeError, RuntimeError):
        # Layer attributes may not exist or device operations may fail.
        pass


def _register_litert_decompositions(torch, litert_torch):
    """Register decompositions for ops that litert_torch cannot lower.

    Covers two categories:
    1. MPS-specific ops on Apple Silicon (e.g. MPS SDPA variant).
    2. Ops whose litert_torch lowering doesn't accept all kwargs
       (e.g. ``aten.mean.dim`` with ``dtype``).
    """
    from litert_torch.fx_infra import decomp as litert_decomp

    pre_convert = litert_decomp.pre_convert_decomp()

    # --- MPS SDPA decomposition ---
    mps_sdpa = getattr(
        torch.ops.aten,
        "_scaled_dot_product_attention_math_for_mps",
        None,
    )
    if mps_sdpa is not None:
        mps_sdpa_default = getattr(mps_sdpa, "default", None)
        if mps_sdpa_default is not None and mps_sdpa_default not in pre_convert:
            non_mps_sdpa = getattr(
                torch.ops.aten._scaled_dot_product_attention_math,
                "default",
                None,
            )
            if non_mps_sdpa is not None:
                core_decomps = torch._decomp.core_aten_decompositions()
                if non_mps_sdpa in core_decomps:
                    litert_decomp.add_pre_convert_decomp(
                        mps_sdpa_default, core_decomps[non_mps_sdpa]
                    )

    # --- aten.mean.dim decomposition (strip dtype kwarg) ---
    # litert_torch's lowering for aten.mean.dim does not accept the
    # ``dtype`` keyword argument. We register a decomposition that
    # implements mean via sum/division when dtype is specified, avoiding
    # a recursive call to the same op.
    mean_dim_op = torch.ops.aten.mean.dim
    if mean_dim_op not in pre_convert:

        def _mean_dim_with_dtype(self, dim, keepdim=False, *, dtype=None):
            if dtype is not None:
                self = self.to(dtype)
            # Compute mean via sum / count to avoid recursing into
            # the same decomposition entry.
            curr_dim = dim
            if curr_dim is None:
                curr_dim = list(range(self.ndim))
            elif isinstance(curr_dim, int):
                curr_dim = [curr_dim]
            count = 1
            for d in curr_dim:
                count *= self.shape[d]
            return (
                torch.ops.aten.sum.dim_IntList(self, curr_dim, keepdim=keepdim)
                / count
            )

        litert_decomp.add_pre_convert_decomp(mean_dim_op, _mean_dim_with_dtype)

    # --- aten.repeat_interleave decomposition ---
    # litert_torch does not have a lowering for
    # aten.repeat_interleave.Tensor. Decompose it into
    # unsqueeze + expand + reshape which are portable ops.
    repeat_interleave_op = getattr(
        torch.ops.aten.repeat_interleave, "Tensor", None
    )
    if (
        repeat_interleave_op is not None
        and repeat_interleave_op not in pre_convert
    ):

        def _repeat_interleave_decomp(repeats, output_size=None):
            # This overload takes a 1-D repeats tensor and returns
            # indices where each index i is repeated repeats[i] times.
            # Example: [2, 1] -> [0, 0, 1].
            # Use cumsum and searchsorted to compute indices portably.
            if output_size is None:
                output_size = torch.ops.aten.sum.default(repeats)
            # cumsum of repeats gives boundaries: [2, 1] -> [2, 3]
            boundaries = torch.ops.aten.cumsum.default(
                repeats, dim=0, dtype=repeats.dtype
            )
            # Create output indices: arange(output_size) gives [0, 1, 2]
            out_indices = torch.ops.aten.arange.start_step(
                0, output_size, 1, dtype=repeats.dtype, device=repeats.device
            )
            # searchsorted: for each out_indices[i], find which repeats
            # bucket it falls into. [0, 1, 2] with boundaries [2, 3]
            # -> [0, 0, 1] (0 and 1 < 2, so bucket 0; 2 >= 2 and < 3, so 1)
            return torch.ops.aten.searchsorted.default(
                boundaries, out_indices, right=False
            )

        # Only register if not already registered
        litert_decomp.add_pre_convert_decomp(
            repeat_interleave_op, _repeat_interleave_decomp
        )

    # Also handle the self_int overload (scalar repeats along a dim)
    repeat_interleave_self_int = getattr(
        torch.ops.aten.repeat_interleave, "self_int", None
    )
    if (
        repeat_interleave_self_int is not None
        and repeat_interleave_self_int not in pre_convert
    ):

        def _repeat_interleave_self_int_decomp(
            self, repeats, dim=None, *, output_size=None
        ):
            if dim is None:
                self = self.flatten()
                dim = 0
            if dim < 0:
                dim = self.ndim + dim
            # unsqueeze after dim, expand, then flatten the two dims
            x = self.unsqueeze(dim + 1)
            expand_shape = [-1] * x.ndim
            expand_shape[dim + 1] = repeats
            x = x.expand(expand_shape)
            shape = list(self.shape)
            shape[dim] = shape[dim] * repeats
            return x.reshape(shape)

        litert_decomp.add_pre_convert_decomp(
            repeat_interleave_self_int, _repeat_interleave_self_int_decomp
        )


def _patch_vhlo_target_version():
    """Patch VHLO serialization for TFLite converter compatibility.

    Addresses two issues in the litert_torch → TFLite conversion path:

    1. **StableHLO version skew**: litert_torch serializes VHLO bytecode
       targeting StableHLO v1.13.0+ (``WEEK_12``), which introduces v2
       variants of element-wise ops (``tanh_v2``, ``logistic_v2``, etc.).
       The TFLite converter only supports StableHLO up to ~v1.9.1.

    2. **Unsupported ``optimization_barrier`` ops**: litert_torch's
       embedding composite pass inserts ``stablehlo.optimization_barrier``
       ops that the TFLite converter cannot handle.

    This function monkey-patches ``MlirLowered.module_bytecode_vhlo`` to
    strip ``optimization_barrier`` ops and serialize with the minimum
    StableHLO version.
    """
    try:
        from litert_torch.odml_torch import export as _odml_export

        MlirLowered = _odml_export.MlirLowered
        if getattr(MlirLowered, "_keras_vhlo_patched", False):
            return

        _serialize = _odml_export.serialize_portable_artifact
        _stablehlo = _odml_export.stablehlo
        _min_version = _stablehlo.get_minimum_version()

        @property
        def _patched_module_bytecode_vhlo(self):
            _remove_optimization_barriers(self.module)
            return _serialize(self.module_bytecode, _min_version)

        MlirLowered.module_bytecode_vhlo = _patched_module_bytecode_vhlo
        MlirLowered._keras_vhlo_patched = True
    except (ImportError, AttributeError):
        # VHLO patching is best-effort. If litert_torch internals are
        # unavailable, the conversion will proceed without this optimization.
        pass


def _remove_optimization_barriers(module):
    """Remove ``stablehlo.optimization_barrier`` ops from an MLIR module.

    ``optimization_barrier`` is a compiler hint that prevents reordering;
    it has no runtime effect.  Each output is replaced with its
    corresponding input before the op is erased.
    """

    def _walk_and_remove(region):
        for block in region:
            to_erase = []
            for op in block:
                # Recurse into nested regions (e.g. while, if).
                for nested in op.regions:
                    _walk_and_remove(nested)
                if op.name == "stablehlo.optimization_barrier":
                    for result, operand in zip(op.results, op.operands):
                        result.replace_all_uses_with(operand)
                    to_erase.append(op)
            for op in reversed(to_erase):
                op.erase()

    try:
        for op in module.body.operations:
            for region in op.regions:
                _walk_and_remove(region)
    except (AttributeError, RuntimeError):
        # MLIR operations may not have expected structure. This is a
        # best-effort optimization; conversion can proceed without it.
        pass
