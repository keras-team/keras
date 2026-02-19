import warnings

from keras.src import backend
from keras.src import tree
from keras.src.export.export_utils import convert_spec_to_tensor
from keras.src.export.export_utils import get_input_signature
from keras.src.utils import io_utils


def export_torch(
    model,
    filepath,
    input_signature=None,
    verbose=None,
    **kwargs,
):
    """Export the model as a PyTorch ExportedProgram artifact for inference.

    This method exports the model to a PyTorch `.pt2` file using
    `torch.export.export`. The exported model can be loaded in any PyTorch
    environment via `torch.export.load()` and used for inference without
    needing the original Keras code. The `.pt2` format is also compatible
    with downstream tools like `ai-edge-torch` for conversion to LiteRT.

    Args:
        model: The Keras model to export.
        filepath: `str` or `pathlib.Path` object. The path to save the
            exported artifact. Must end with `.pt2`.
        input_signature: Optional input signature specification. If
            `None`, it will be inferred from the model.
        verbose: `bool`. Whether to print a message after export.
            Defaults to `True`.
        **kwargs: Additional keyword arguments passed to
            `torch.export.export` (e.g., `strict`).

    **Note:** This feature requires the PyTorch backend
    (`KERAS_BACKEND=torch`).

    **Note:** `torch.export` captures the full computation graph. Shapes
    are fixed at export time based on the input signature. To export with
    a specific batch size, provide an explicit `input_signature`.

    Example:

    ```python
    # Export the model as a PyTorch ExportedProgram
    model.export("path/to/model.pt2", format="torch")

    # Load the artifact in a different process/environment
    import torch
    loaded_program = torch.export.load("path/to/model.pt2")
    output = loaded_program.module()(torch.randn(1, 10))
    ```
    """
    exporter = TorchExporter(
        model=model,
        input_signature=input_signature,
        **kwargs,
    )
    exporter.export(filepath)
    actual_verbose = verbose if verbose is not None else True
    if actual_verbose:
        io_utils.print_msg(f"Saved PyTorch ExportedProgram at '{filepath}'.")


class TorchExporter:
    """Exporter for the PyTorch ExportedProgram format.

    This class handles the conversion of Keras models to PyTorch's
    `ExportedProgram` format using `torch.export.export` and generates a
    `.pt2` model file. The exported model contains the model's forward pass
    and all learned parameters, and can be loaded via
    `torch.export.load()`. It is also compatible with `ai-edge-torch` for
    further conversion to LiteRT.
    """

    def __init__(
        self,
        model,
        input_signature=None,
        **kwargs,
    ):
        """Initialize the PyTorch exporter.

        Args:
            model: The Keras model to export.
            input_signature: Input signature specification (e.g.,
                `keras.InputSpec` or list of `keras.InputSpec`).
            **kwargs: Additional parameters passed to
                `torch.export.export`.
        """
        if backend.backend() != "torch":
            raise RuntimeError(
                "`export_torch` is only compatible with the PyTorch backend. "
                f"Current backend: '{backend.backend()}'."
            )

        self.model = model
        self.input_signature = input_signature
        self.kwargs = kwargs

    def export(self, filepath):
        """Export the Keras model to a PyTorch ExportedProgram file.

        Args:
            filepath: Output path for the exported model. Must end with
                `.pt2`.

        Returns:
            Path to the exported model.
        """
        import torch

        filepath = str(filepath)
        if not filepath.endswith(".pt2"):
            raise ValueError(
                "The PyTorch export requires the filepath to end with "
                f"'.pt2'. Got: {filepath}"
            )

        # 1. Resolve / infer input signature
        if self.input_signature is None:
            self.input_signature = get_input_signature(self.model)

        # 2. Create sample inputs from signature
        #    (replace None dimensions with concrete value)
        sample_inputs = tree.map_structure(
            lambda x: convert_spec_to_tensor(x, replace_none_number=1),
            self.input_signature,
        )
        sample_inputs = tuple(sample_inputs)

        # 3. Move sample inputs to same device as model
        device = self._get_model_device()
        if device is not None:
            sample_inputs = tuple(
                t.to(device) if hasattr(t, "to") else t for t in sample_inputs
            )

        # 4. Put model in eval mode for inference
        if hasattr(self.model, "eval"):
            self.model.eval()

        # 5. Configure torch.export parameters
        export_kwargs = self._get_export_kwargs()

        # 6. Export and save the model
        with warnings.catch_warnings():
            # Suppress Keras internal warnings about submodule registration
            warnings.filterwarnings(
                "ignore",
                message=r".*not properly registered as a submodule.*",
            )

            try:
                exported_program = torch.export.export(
                    self.model,
                    sample_inputs,
                    **export_kwargs,
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to export model to PyTorch format. "
                    f"Common causes: unsupported operations, data-dependent "
                    f"control flow, or dynamic shapes. Original error: {e}"
                ) from e

        torch.export.save(exported_program, filepath)
        return filepath

    def _get_model_device(self):
        """Detect the device of model parameters.

        Returns:
            torch.device or None if device cannot be determined.
        """
        try:
            for v in self.model.variables:
                if hasattr(v, "value") and hasattr(v.value, "device"):
                    return v.value.device
        except (AttributeError, TypeError):
            pass
        return None

    def _get_export_kwargs(self):
        """Extract and configure torch.export parameters.

        Returns:
            dict: Parameters for torch.export.export.
        """
        export_kwargs = {}
        if "strict" in self.kwargs:
            export_kwargs["strict"] = self.kwargs["strict"]
        else:
            # Default to non-strict mode: Keras models often use patterns
            # (e.g., custom layers with Python control flow) that strict
            # mode rejects. Non-strict mode allows these while still
            # capturing the core computation graph.
            export_kwargs["strict"] = False
        return export_kwargs
