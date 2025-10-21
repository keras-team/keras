import logging
import os

from keras.src import tree
from keras.src.utils import io_utils
from keras.src.utils.module_utils import litert
from keras.src.utils.module_utils import tensorflow as tf


def export_litert(
    model,
    filepath,
    verbose=True,
    input_signature=None,
    aot_compile_targets=None,
    **kwargs,
):
    """Export the model as a LiteRT artifact for inference.

    Args:
        model: The Keras model to export.
        filepath: The path to save the exported artifact.
        verbose: `bool`. Whether to print a message during export. Defaults to
            `None`, which uses the default value set by different backends and
            formats.
        input_signature: Optional input signature specification. If
            `None`, it will be inferred.
        aot_compile_targets: Optional list of LiteRT targets for AOT
        compilation.
        **kwargs: Additional keyword arguments passed to the exporter.
    """

    exporter = LiteRTExporter(
        model=model,
        input_signature=input_signature,
        verbose=verbose,
        aot_compile_targets=aot_compile_targets,
        **kwargs,
    )
    exporter.export(filepath)
    if verbose:
        io_utils.print_msg(f"Saved artifact at '{filepath}'.")


class LiteRTExporter:
    """
    Exporter for the LiteRT (TFLite) format that creates a single,
    callable signature for `model.call`.
    """

    def __init__(
        self,
        model,
        input_signature=None,
        verbose=False,
        aot_compile_targets=None,
        **kwargs,
    ):
        """Initialize the LiteRT exporter.

        Args:
            model: The Keras model to export
            input_signature: Input signature specification
            verbose: Whether to print progress messages during export.
            aot_compile_targets: List of LiteRT targets for AOT compilation
            **kwargs: Additional export parameters
        """
        self.model = model
        self.input_signature = input_signature
        self.verbose = verbose
        self.aot_compile_targets = aot_compile_targets
        self.kwargs = kwargs

    def export(self, filepath):
        """Exports the Keras model to a TFLite file and optionally performs AOT
        compilation.

        Args:
            filepath: Output path for the exported model

        Returns:
            Path to exported model or compiled models if AOT compilation is
            performed
        """
        if self.verbose:
            io_utils.print_msg("Starting LiteRT export...")

        # 1. Ensure the model is built by calling it if necessary
        self._ensure_model_built()

        # 2. Resolve / infer input signature
        if self.input_signature is None:
            if self.verbose:
                io_utils.print_msg("Inferring input signature from model.")
            from keras.src.export.export_utils import get_input_signature

            self.input_signature = get_input_signature(self.model)

        # 3. Convert the model to TFLite.
        tflite_model = self._convert_to_tflite(self.input_signature)

        if self.verbose:
            final_size_mb = len(tflite_model) / (1024 * 1024)
            io_utils.print_msg(
                f"TFLite model converted successfully. Size: "
                f"{final_size_mb:.2f} MB"
            )

        # 4. Save the initial TFLite model to the specified file path.
        assert filepath.endswith(".tflite"), (
            "The LiteRT export requires the filepath to end with '.tflite'. "
            f"Got: {filepath}"
        )

        with open(filepath, "wb") as f:
            f.write(tflite_model)

        if self.verbose:
            io_utils.print_msg(f"TFLite model saved to {filepath}")

        # 5. Perform AOT compilation if targets are specified and LiteRT is
        # available
        compiled_models = None
        if self.aot_compile_targets and litert.available:
            if self.verbose:
                io_utils.print_msg(
                    "Performing AOT compilation for LiteRT targets..."
                )
            compiled_models = self._aot_compile(filepath)
        elif self.aot_compile_targets and not litert.available:
            logging.warning(
                "AOT compilation requested but LiteRT is not available. "
                "Skipping AOT compilation."
            )

        if self.verbose:
            io_utils.print_msg(
                f"LiteRT export completed. Base model: {filepath}"
            )
            if compiled_models:
                io_utils.print_msg(
                    f"AOT compiled models: {len(compiled_models.models)} "
                    "variants"
                )

        return compiled_models if compiled_models else filepath

    def _ensure_model_built(self):
        """
        Ensures the model is built before conversion.

        For models that are not yet built, this attempts to build them
        using the input signature or model.inputs.
        """
        if self.model.built:
            return

        if self.verbose:
            io_utils.print_msg("Building model before conversion...")

        try:
            # Try to build using input_signature if available
            if self.input_signature:
                input_shapes = tree.map_structure(
                    lambda spec: spec.shape, self.input_signature
                )
                self.model.build(input_shapes)
            # Fall back to model.inputs for Functional/Sequential models
            elif hasattr(self.model, "inputs") and self.model.inputs:
                input_shapes = [inp.shape for inp in self.model.inputs]
                if len(input_shapes) == 1:
                    self.model.build(input_shapes[0])
                else:
                    self.model.build(input_shapes)
            else:
                raise ValueError(
                    "Cannot export model to the litert format as the "
                    "input_signature could not be inferred. Either pass an "
                    "`input_signature` to `model.export()` or ensure that the "
                    "model is already built (called once on real inputs)."
                )

            if self.verbose:
                io_utils.print_msg("Model built successfully.")

        except Exception as e:
            if self.verbose:
                io_utils.print_msg(f"Error building model: {e}")
            raise ValueError(
                f"Failed to build model: {e}. Please ensure the model is "
                "properly defined or provide an input_signature."
            )

    def _convert_to_tflite(self, input_signature):
        """Converts the Keras model to a TFLite model."""
        is_sequential = isinstance(self.model, tf.keras.Sequential)

        # Try direct conversion first for all models
        try:
            if self.verbose:
                model_type = "Sequential" if is_sequential else "Functional"
                io_utils.print_msg(
                    f"{model_type} model detected. Trying direct conversion..."
                )

            converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,
                tf.lite.OpsSet.SELECT_TF_OPS,
            ]
            converter.experimental_enable_resource_variables = False
            tflite_model = converter.convert()

            if self.verbose:
                io_utils.print_msg("Direct conversion successful.")
            return tflite_model

        except Exception as direct_error:
            if self.verbose:
                model_type = "Sequential" if is_sequential else "Functional"
                io_utils.print_msg(
                    f"Direct conversion failed for {model_type} model: "
                    f"{direct_error}"
                )
                io_utils.print_msg(
                    "Falling back to wrapper-based conversion..."
                )

            return self._convert_with_wrapper(input_signature)

    def _convert_with_wrapper(self, input_signature):
        """Converts the model to TFLite using the tf.Module wrapper."""

        # Define the wrapper class dynamically to avoid module-level
        # tf.Module inheritance
        class KerasModelWrapper(tf.Module):
            """
            A tf.Module wrapper for a Keras model.

            This wrapper is designed to be a clean, serializable interface
            for TFLite conversion. It holds the Keras model and exposes a
            single `__call__` method that is decorated with `tf.function`.
            Crucially, it also ensures all variables from the Keras model
            are tracked by the SavedModel format, which is key to including
            them in the final TFLite model.
            """

            def __init__(self, model):
                super().__init__()
                # Store the model reference in a way that TensorFlow won't
                # try to track it. This prevents the _DictWrapper error during
                # SavedModel serialization
                object.__setattr__(self, "_model", model)

                # Track all variables from the Keras model using proper
                # tf.Module methods. This ensures proper variable handling for
                # stateful layers like BatchNorm
                with self.name_scope:
                    for i, var in enumerate(model.variables):
                        # Use a different attribute name to avoid conflicts with
                        # tf.Module's variables property
                        setattr(self, f"model_var_{i}", var)

            @tf.function
            def __call__(self, *args, **kwargs):
                """The single entry point for the exported model."""
                # Handle both single and multi-input cases
                if args and not kwargs:
                    # Called with positional arguments
                    if len(args) == 1:
                        return self._model(args[0])
                    else:
                        return self._model(list(args))
                elif kwargs and not args:
                    # Called with keyword arguments
                    if len(kwargs) == 1 and "inputs" in kwargs:
                        # Single input case
                        return self._model(kwargs["inputs"])
                    else:
                        # Multi-input case - convert to list/dict format
                        # expected by model
                        if (
                            hasattr(self._model, "inputs")
                            and len(self._model.inputs) > 1
                        ):
                            # Multi-input functional model
                            input_list = []
                            missing_inputs = []
                            for input_layer in self._model.inputs:
                                input_name = input_layer.name
                                if input_name in kwargs:
                                    input_list.append(kwargs[input_name])
                                else:
                                    missing_inputs.append(input_name)

                            if missing_inputs:
                                available = list(kwargs.keys())
                                raise ValueError(
                                    f"Missing required inputs for multi-input "
                                    f"model: {missing_inputs}. "
                                    f"Available kwargs: {available}. "
                                    f"Please provide all inputs by name."
                                )

                            return self._model(input_list)
                        else:
                            # Single input model called with named arguments
                            return self._model(list(kwargs.values())[0])
                else:
                    # Fallback to original call
                    return self._model(*args, **kwargs)

        # 1. Wrap the Keras model in our clean tf.Module.
        wrapper = KerasModelWrapper(self.model)

        # 2. Get a concrete function from the wrapper.
        if not isinstance(input_signature, (list, tuple)):
            input_signature = [input_signature]

        from keras.src.export.export_utils import make_tf_tensor_spec

        tensor_specs = [make_tf_tensor_spec(spec) for spec in input_signature]

        # Pass tensor specs as positional arguments to get the concrete
        # function.
        concrete_func = wrapper.__call__.get_concrete_function(*tensor_specs)

        # 3. Convert from the concrete function.
        if self.verbose:
            io_utils.print_msg(
                "Converting concrete function to TFLite format..."
            )

        # Try multiple conversion strategies for better inference compatibility
        conversion_strategies = [
            {
                "experimental_enable_resource_variables": False,
                "name": "without resource variables",
            },
            {
                "experimental_enable_resource_variables": True,
                "name": "with resource variables",
            },
        ]

        for strategy in conversion_strategies:
            try:
                converter = tf.lite.TFLiteConverter.from_concrete_functions(
                    [concrete_func], trackable_obj=wrapper
                )
                converter.target_spec.supported_ops = [
                    tf.lite.OpsSet.TFLITE_BUILTINS,
                    tf.lite.OpsSet.SELECT_TF_OPS,
                ]
                converter.experimental_enable_resource_variables = strategy[
                    "experimental_enable_resource_variables"
                ]

                if self.verbose:
                    io_utils.print_msg(
                        f"Trying conversion {strategy['name']}..."
                    )

                tflite_model = converter.convert()

                if self.verbose:
                    io_utils.print_msg(
                        f"Conversion successful {strategy['name']}!"
                    )

                return tflite_model

            except Exception as e:
                if self.verbose:
                    io_utils.print_msg(
                        f"Conversion failed {strategy['name']}: {e}"
                    )
                continue

        # If all strategies fail, raise the last error
        raise RuntimeError(
            "All conversion strategies failed for wrapper-based conversion"
        )

    def _aot_compile(self, tflite_filepath):
        """Performs AOT compilation using LiteRT."""
        if not litert.available:
            raise RuntimeError("LiteRT is not available for AOT compilation")

        try:
            # Create a LiteRT model from the TFLite file
            litert_model = litert.python.aot.core.types.Model.create_from_path(
                tflite_filepath
            )

            # Determine output directory
            base_dir = os.path.dirname(tflite_filepath)
            model_name = os.path.splitext(os.path.basename(tflite_filepath))[0]
            output_dir = os.path.join(base_dir, f"{model_name}_compiled")

            if self.verbose:
                io_utils.print_msg(
                    f"AOT compiling for targets: {self.aot_compile_targets}"
                )
                io_utils.print_msg(f"Output directory: {output_dir}")

            # Perform AOT compilation
            result = litert.python.aot.aot_compile(
                input_model=litert_model,
                output_dir=output_dir,
                target=self.aot_compile_targets,
                keep_going=True,  # Continue even if some targets fail
            )

            if self.verbose:
                io_utils.print_msg(
                    f"AOT compilation completed: {len(result.models)} "
                    f"successful, {len(result.failed_backends)} failed"
                )
                if result.failed_backends:
                    for backend, error in result.failed_backends:
                        io_utils.print_msg(
                            f"  Failed: {backend.id()} - {error}"
                        )

                # Print compilation report if available
                try:
                    report = result.compilation_report()
                    if report:
                        io_utils.print_msg("Compilation Report:")
                        io_utils.print_msg(report)
                except Exception:
                    pass

            return result

        except Exception as e:
            if self.verbose:
                io_utils.print_msg(f"AOT compilation failed: {e}")
                import traceback

                traceback.print_exc()
            raise RuntimeError(f"AOT compilation failed: {e}")

    def _get_available_litert_targets(self):
        """Get available LiteRT targets for AOT compilation."""
        if not litert.available:
            return []

        try:
            # Get all registered targets
            targets = (
                litert.python.aot.vendors.import_vendor.AllRegisteredTarget()
            )
            return targets if isinstance(targets, list) else [targets]
        except Exception as e:
            if self.verbose:
                io_utils.print_msg(f"Failed to get available targets: {e}")
            return []

    @classmethod
    def export_with_aot(
        cls, model, filepath, targets=None, verbose=True, **kwargs
    ):
        """
        Convenience method to export a Keras model with AOT compilation.

        Args:
            model: Keras model to export
            filepath: Output file path
            targets: List of LiteRT targets for AOT compilation (e.g.,
            ['qualcomm', 'mediatek'])
            verbose: Whether to print verbose output
            **kwargs: Additional arguments for the exporter

        Returns:
            CompilationResult if AOT compilation is performed, otherwise the
            filepath
        """
        exporter = cls(
            model=model, verbose=verbose, aot_compile_targets=targets, **kwargs
        )
        return exporter.export(filepath)

    @classmethod
    def get_available_targets(cls):
        """Get list of available LiteRT AOT compilation targets."""
        if not litert.available:
            return []

        dummy_exporter = cls(model=None)
        return dummy_exporter._get_available_litert_targets()
