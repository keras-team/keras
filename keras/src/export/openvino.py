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

    actual_verbose = verbose
    if actual_verbose is None:
        actual_verbose = True

    if backend.backend() == "openvino":
        if input_signature is None:
            input_signature = get_input_signature(model)
        if isinstance(input_signature, list) and len(input_signature) == 1:
            input_signature = input_signature[0]
        sample_inputs = tree.map_structure(
            lambda x: convert_spec_to_tensor(x, replace_none_number=1),
            input_signature,
        )
        sample_inputs = model(sample_inputs)

    elif backend.backend() in ("tensorflow", "jax", "torch"):
        if backend.backend() in ("torch", "jax"):
            import shutil
            import tempfile

            with tempfile.TemporaryDirectory() as temp_dir:
                model.export(temp_dir, format="tf_saved_model")
                ov_model = ov.convert_model(temp_dir)
                shutil.rmtree(temp_dir)
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
