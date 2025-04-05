# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pathlib
from typing import Any

from openvino import Model  # pylint: disable=no-name-in-module,import-error
from openvino.tools.ovc.cli_parser import get_all_cli_parser
from openvino.tools.ovc.convert_impl import _convert
from openvino.tools.ovc.logger import get_logger_state, restore_logger_state


def convert_model(
        input_model: [str, pathlib.Path, Any, list],
        # TODO: Instead of list just accept arbitrary number of positional arguments

        # Framework-agnostic parameters
        input: [list, dict, str] = None,
        output: [str, list] = None,
        example_input: Any = None,
        extension: [str, pathlib.Path, list, Any] = None,
        verbose: bool = False,
        share_weights: bool = True,
) -> Model:
    """
    Converts the model from original framework to OpenVino Model.

    Framework-agnostic parameters:
        :param input_model:
            Model object in original framework (PyTorch, TensorFlow) or path to model file.

            Supported formats of input model:

            PyTorch
            torch.nn.Module
            torch.jit.ScriptModule
            torch.jit.ScriptFunction
            torch.export.ExportedProgram

            TF
            tf.compat.v1.Graph
            tf.compat.v1.GraphDef
            tf.compat.v1.wrap_function
            tf.compat.v1.session

            TF2 / Keras
            tf.keras.Model
            tf.keras.layers.Layer
            tf.function
            tf.Module
            tf.train.checkpoint

            PaddlePaddle
            paddle.hapi.model.Model
            paddle.fluid.dygraph.layers.Layer
            paddle.fluid.executor.Executor

        :param input:
            Information of model input required for model conversion.
            Input can be set by a list of tuples or a dictionary. Each tuple can contain optionally input name (string),
            input type (ov.Type, numpy.dtype) or input shape (ov.Shape, ov.PartialShape, list, tuple).
            Example: input=("op_name", PartialShape([-1, 3, 100, 100]), ov.Type.f32).
            Alternatively input can be set by a dictionary, where key - input name,
            value - tuple with input parameters (shape or type).
            Example 1: input={"op_name_1": ([1, 2, 3], ov.Type.f32), "op_name_2": ov.Type.i32}
            Example 2: input=[("op_name_1", [1, 2, 3], ov.Type.f32), ("op_name_2", ov.Type.i32)]
            Example 3: input=[([1, 2, 3], ov.Type.f32), ov.Type.i32]
            The order of inputs in converted model will match the order of specified inputs.
            If data type is not specified explicitly data type is taken from the original node data type.

        :param output:
            One or more comma-separated model outputs to be preserved in the converted model. Other outputs are removed.
            If `output` parameter is not specified then all outputs from the original model are preserved.
            Do not add :0 to the names for TensorFlow. The order of outputs in the converted model is the same as
            the order of specified names. Example: output="out_1", or output=["out_1", "out_2"].
            For PaddlePaddle model represented as a Python object, you can specify outputs as a PaddlePaddle Python
            objects or a list of such objects.
        :param example_input:
            Sample of model input in original framework.
            For PyTorch it can be torch.Tensor.
            For Tensorflow it can be tf.Tensor or numpy.ndarray.
            For PaddlePaddle it can be Paddle Variable.
        :param extension:
            Paths to libraries (.so or .dll) with extensions, comma-separated
            list of paths, objects derived from BaseExtension class or lists of
            objects.
        :param verbose:
            Print detailed information about conversion. The detailed information is logged via standard logging library.
            The log level can be changed by setting the log level using logging library.
            Example:
                import logging
                logging.getLogger().setLevel(logging.DEBUG)
        :param share_weights:
            Reuse weights allocated in the original model. If input model is in file,
            then mmap is used to allocate weights directly from file. If input model is
            runtime object, then original memory regions allocated in the original model
            are reused for weights in the converted model.

    Returns:
        openvino.Model
    """
    params = locals()
    logger_state = get_logger_state()
    cli_parser = get_all_cli_parser()
    ov_model, _ = _convert(cli_parser, params, True)
    restore_logger_state(logger_state)
    return ov_model
