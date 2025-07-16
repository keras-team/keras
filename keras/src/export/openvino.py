from keras.src import backend
from keras.src import tree
from keras.src.export.export_utils import convert_spec_to_tensor
from keras.src.export.export_utils import get_input_signature
from keras.src.utils import io_utils


def export_openvino(
    model, filepath, verbose=None, input_signature=None, **kwargs
):
    import os

    import openvino as ov
    from openvino.runtime import opset14 as ov_opset

    from keras.src.backend.openvino.core import OPENVINO_DTYPES
    from keras.src.backend.openvino.core import OpenVINOKerasTensor

    actual_verbose = verbose
    if actual_verbose is None:
        actual_verbose = True

    if backend.backend() == "openvino":

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

        if input_signature is None:
            input_signature = get_input_signature(model)
        if isinstance(input_signature, list) and len(input_signature) == 1:
            input_signature = input_signature[0]
        sample_inputs = tree.map_structure(
            lambda x: convert_spec_to_tensor(x, replace_none_number=1),
            input_signature,
        )
        params = parameterize_inputs(sample_inputs)
        outputs = model(params)
        parameters = [p.output.get_node() for p in tree.flatten(params)]
        results = [ov_opset.result(r.output) for r in tree.flatten(outputs)]
        ov_model = ov.Model(results=results, parameters=parameters)
        for param in ov_model.inputs:
            rank = len(param.get_partial_shape())
            dynamic_shape = ov.PartialShape([-1] * rank)
            param.get_node().set_partial_shape(dynamic_shape)
    elif backend.backend() in ("tensorflow", "jax", "torch"):
        if backend.backend() in ("torch", "jax", "tensorflow"):
            import tempfile

            with tempfile.TemporaryDirectory() as temp_dir:
                model.export(temp_dir, format="tf_saved_model")
                ov_model = ov.convert_model(temp_dir)
        else:
            ov_model = ov.convert_model(model)
    else:
        raise NotImplementedError(
            "`export_openvino` is only compatible with OpenVINO, "
            "TensorFlow, JAX and Torch backends."
        )

    xml_path = os.path.join(filepath, f"{model.name}.xml")
    ov.serialize(ov_model, xml_path)

    if actual_verbose:
        io_utils.print_msg(f"Saved OpenVINO IR at '{filepath}'.")
