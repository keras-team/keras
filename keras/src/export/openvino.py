import warnings

from keras.src import backend
from keras.src import tree
from keras.src.export.export_utils import convert_spec_to_tensor
from keras.src.export.export_utils import get_input_signature
from keras.src.export.export_utils import make_tf_tensor_spec
from keras.src.export.saved_model import DEFAULT_ENDPOINT_NAME
from keras.src.export.saved_model import ExportArchive
from keras.src.utils import io_utils


def export_openvino(
    model, filepath, verbose=None, input_signature=None, **kwargs
):
    """Export the model as an OpenVINO IR artifact for inference.

    This method exports the model to the OpenVINO IR format,
    which includes two files:
    a `.xml` file containing the model structure and a `.bin` file
    containing the weights.
    The exported model contains only the forward pass
    (i.e., the model's `call()` method), and can be deployed with the
    OpenVINO Runtime for fast inference on CPU and other Intel hardware.

    Args:
        filepath: `str` or `pathlib.Path`. Path to the output `.xml` file.
        The corresponding `.bin` file will be saved alongside it.
        verbose: Optional `bool`. Whether to print a confirmation message
        after export. If `None`, it uses the default verbosity configured
        by the backend.
        input_signature: Optional. Specifies the shape and dtype of the
        model inputs. If not provided, it will be inferred.
        **kwargs: Additional keyword arguments.

     Example:

    ```python
    import keras

    # Define or load a Keras model
    model = keras.models.Sequential([
        keras.layers.Input(shape=(128,)),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(10)
    ])

    # Export to OpenVINO IR
    model.export("model.xml", format="openvino")
    ```
    """
    assert filepath.endswith(".xml"), (
        "The OpenVINO export requires the filepath to end with '.xml'. "
        f"Got: {filepath}"
    )

    import openvino as ov
    import openvino.opset14 as ov_opset

    from keras.src.backend.openvino.core import OPENVINO_DTYPES
    from keras.src.backend.openvino.core import OpenVINOKerasTensor

    actual_verbose = verbose if verbose is not None else True

    if input_signature is None:
        input_signature = get_input_signature(model)

    if backend.backend() == "openvino":
        import inspect

        def parameterize_inputs(inputs, prefix=""):
            if isinstance(inputs, (list, tuple)):
                return [
                    parameterize_inputs(e, f"{prefix}{i}")
                    for i, e in enumerate(inputs)
                ]
            elif isinstance(inputs, dict):
                return {k: parameterize_inputs(v, k) for k, v in inputs.items()}
            elif isinstance(inputs, OpenVINOKerasTensor):
                ov_type = OPENVINO_DTYPES[str(inputs.dtype)]
                ov_shape = list(inputs.shape)
                param = ov_opset.parameter(shape=ov_shape, dtype=ov_type)
                param.set_friendly_name(prefix)
                return OpenVINOKerasTensor(param.output(0))
            else:
                raise TypeError(f"Unknown input type: {type(inputs)}")

        if isinstance(input_signature, list) and len(input_signature) == 1:
            input_signature = input_signature[0]

        sample_inputs = tree.map_structure(
            lambda x: convert_spec_to_tensor(x, replace_none_number=1),
            input_signature,
        )
        params = parameterize_inputs(sample_inputs)
        signature = inspect.signature(model.call)
        if len(signature.parameters) > 1 and isinstance(params, (list, tuple)):
            outputs = model(*params)
        else:
            outputs = model(params)
        parameters = [p.output.get_node() for p in tree.flatten(params)]
        results = [ov_opset.result(r.output) for r in tree.flatten(outputs)]
        ov_model = ov.Model(results=results, parameters=parameters)
        flat_specs = tree.flatten(input_signature)
        for ov_input, spec in zip(ov_model.inputs, flat_specs):
            # Respect the dynamic axes from the original input signature.
            dynamic_shape_dims = [
                -1 if dim is None else dim for dim in spec.shape
            ]
            dynamic_shape = ov.PartialShape(dynamic_shape_dims)
            ov_input.get_node().set_partial_shape(dynamic_shape)

    elif backend.backend() in ("tensorflow", "jax"):
        inputs = tree.map_structure(make_tf_tensor_spec, input_signature)
        decorated_fn = get_concrete_fn(model, inputs, **kwargs)
        ov_model = ov.convert_model(decorated_fn)
        set_names(ov_model, inputs)
    elif backend.backend() == "torch":
        import torch

        sample_inputs = tree.map_structure(
            lambda x: convert_spec_to_tensor(x, replace_none_number=1),
            input_signature,
        )
        sample_inputs = tuple(sample_inputs)
        if hasattr(model, "eval"):
            model.eval()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
            traced = torch.jit.trace(model, sample_inputs)
            ov_model = ov.convert_model(traced)
            set_names(ov_model, sample_inputs)
    else:
        raise NotImplementedError(
            "`export_openvino` is only compatible with OpenVINO, "
            "TensorFlow, JAX and Torch backends."
        )

    ov.serialize(ov_model, filepath)

    if actual_verbose:
        io_utils.print_msg(f"Saved OpenVINO IR at '{filepath}'.")


def collect_names(structure):
    if isinstance(structure, dict):
        for k, v in structure.items():
            if isinstance(v, (dict, list, tuple)):
                yield from collect_names(v)
            else:
                yield k
    elif isinstance(structure, (list, tuple)):
        for v in structure:
            yield from collect_names(v)
    else:
        if hasattr(structure, "name") and structure.name:
            yield structure.name
        else:
            yield "input"


def set_names(model, inputs):
    names = list(collect_names(inputs))
    for ov_input, name in zip(model.inputs, names):
        ov_input.get_node().set_friendly_name(name)
        ov_input.tensor.set_names({name})


def _check_jax_kwargs(kwargs):
    kwargs = kwargs.copy()
    if "is_static" not in kwargs:
        kwargs["is_static"] = True
    if "jax2tf_kwargs" not in kwargs:
        kwargs["jax2tf_kwargs"] = {
            "enable_xla": False,
            "native_serialization": False,
        }
    if kwargs["is_static"] is not True:
        raise ValueError(
            "`is_static` must be `True` in `kwargs` when using the jax backend."
        )
    if kwargs["jax2tf_kwargs"]["enable_xla"] is not False:
        raise ValueError(
            "`enable_xla` must be `False` in `kwargs['jax2tf_kwargs']` "
            "when using the jax backend."
        )
    if kwargs["jax2tf_kwargs"]["native_serialization"] is not False:
        raise ValueError(
            "`native_serialization` must be `False` in "
            "`kwargs['jax2tf_kwargs']` when using the jax backend."
        )
    return kwargs


def get_concrete_fn(model, input_signature, **kwargs):
    if backend.backend() == "jax":
        kwargs = _check_jax_kwargs(kwargs)
    export_archive = ExportArchive()
    export_archive.track_and_add_endpoint(
        DEFAULT_ENDPOINT_NAME, model, input_signature, **kwargs
    )
    if backend.backend() == "tensorflow":
        export_archive._filter_and_track_resources()
    return export_archive._get_concrete_fn(DEFAULT_ENDPOINT_NAME)
