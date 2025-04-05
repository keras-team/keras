# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import logging as log
import numpy as np
import os
import sys
from openvino.tools.ovc.error import Error
from typing import Iterable, Union

try:
    import openvino_telemetry as tm
    from openvino_telemetry.backend import backend_ga4
except ImportError:
    import openvino.tools.ovc.telemetry_stub as tm

if sys.version_info < (3, 8):
    import importlib_metadata
else:
    import importlib.metadata as importlib_metadata

dynamic_dimension = np.ma.masked


def refer_to_faq_msg(question_num: int):
    try:
        t = tm.Telemetry()
        t.send_event('ovc', 'error_info', "faq:" + str(question_num))
    except Exception:
        # Telemetry can be not initialized if it is used in MO IR Reader
        pass

    return '\n For more information please refer to Model Conversion API FAQ, question #{0}. ' \
           '(https://docs.openvino.ai/2023.0/openvino_docs_MO_DG_prepare_model_Model_Optimizer_FAQ.html' \
           '?question={0}#question-{0})'.format(question_num)


def get_mo_root_dir():
    """
    Return the absolute path to the Model Conversion API root directory (where mo folder is located)
    :return: path to the MO root directory
    """
    return os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(os.path.realpath(__file__))), os.pardir))


def check_values_equal(val1, val2):
    # This method is needed to check equality of values where some values can be None
    if val1 is None and val2 is None:
        return True
    if val1 is None:
        return False
    if val2 is None:
        return False
    return val1 == val2


np_map_cast = {bool: lambda x: bool_cast(x),
               np.int8: lambda x: np.int8(x),
               np.int16: lambda x: np.int16(x),
               np.int32: lambda x: np.int32(x),
               np.int64: lambda x: np.int64(x),
               np.uint8: lambda x: np.uint8(x),
               np.uint16: lambda x: np.uint16(x),
               np.uint32: lambda x: np.uint32(x),
               np.uint64: lambda x: np.uint64(x),
               np.float16: lambda x: np.float16(x),
               np.float32: lambda x: np.float32(x),
               np.double: lambda x: np.double(x),
               str: lambda x: str(x)}


def bool_cast(x):
    if isinstance(x, str):
        return False if x.lower() in ['false', '0'] else True if x.lower() in ['true', '1'] else 'unknown_boolean_cast'
    else:
        return bool(x)


def mo_array(value: Union[Iterable[Union[float, int]], float, int], dtype=None) -> np.ndarray:
    """
    This function acts in a same way as np.array except for the case when dtype is not provided
    and np.array return fp64 array this function returns fp32 array
    """
    x = np.array(value, dtype=dtype)
    if not isinstance(value, np.ndarray) and x.dtype == np.float64 and dtype != np.float64:
        x = x.astype(np.float32)
    return x


def validate_batch_in_shape(shape, layer_name: str):
    """
    Raises Error #39 if shape is not valid for setting batch size
    Parameters
    ----------
    shape: current shape of layer under validation
    layer_name: name of layer under validation
    """
    if len(shape) == 0 or (shape[0] is not dynamic_dimension and shape[0] not in (-1, 0, 1)):
        raise Error(('The input layer {} has a shape {} defined in the model. \n\n' +
                     'When you use "batch" option, Model Conversion API applies its value to the first ' +
                     'element of the shape if it is equal to -1, 0 or 1. Otherwise, this is the ambiguous ' +
                     'situation - it is not known in advance whether the layer has the batch ' +
                     'dimension or not.\n\n For example, you want to set batch dimension equals 100 ' +
                     'for the input layer "data" with shape (10,34). Although you can not use "batch", ' +
                     'you should pass "input_shape=[100,34]" instead of "batch=100". \n\n' +
                     'You can also specify batch dimension by setting "layout". \n\n')
                    .format(layer_name, shape))


def get_ir_version():
    """
    Default IR version.
    :return: the IR version
    """
    return 11


def import_openvino_tokenizers():
    # extract openvino version
    if importlib.util.find_spec("openvino") is None:
        return False
    try:
        from openvino import get_version
        openvino_version = get_version()
        openvino_available = True
    except ImportError:
        openvino_available = False
    if not openvino_available:
        return False

    if importlib.util.find_spec("openvino_tokenizers") is None:
        return False

    try:
        pip_metadata_version = importlib_metadata.version("openvino")
    except importlib_metadata.PackageNotFoundError:
        pip_metadata_version = False
    try:
        pip_metadata_version = importlib_metadata.version("openvino-nightly")
        is_nightly = True
    except importlib_metadata.PackageNotFoundError:
        is_nightly = False

    try:
        import openvino_tokenizers  # pylint: disable=no-name-in-module,import-error

        openvino_tokenizers._get_factory()
    except RuntimeError:
        tokenizers_version = openvino_tokenizers.__version__

        if tokenizers_version == "0.0.0.0":
            try:
                tokenizers_version = importlib_metadata.version("openvino_tokenizers") or tokenizers_version
            except importlib_metadata.PackageNotFoundError:
                pass
        message = (
            "OpenVINO and OpenVINO Tokenizers versions are not binary compatible.\n"
            f"OpenVINO version:            {openvino_version}\n"
            f"OpenVINO Tokenizers version: {tokenizers_version}\n"
            "First 3 numbers should be the same. Update OpenVINO Tokenizers to compatible version. "
        )
        if not pip_metadata_version:
            message += (
                "For archive installation of OpenVINO try to build OpenVINO Tokenizers from source: "
                "https://github.com/openvinotoolkit/openvino_tokenizers/tree/master?tab=readme-ov-file"
                "#build-and-install-from-source"
            )
            if sys.platform == "linux":
                message += (
                    "\nThe PyPI version of OpenVINO Tokenizers is built on CentOS and may not be compatible with other "
                    "Linux distributions; rebuild OpenVINO Tokenizers from source."
                )
        else:
            message += (
                "It is recommended to use the same day builds for pre-release version. "
                "To install both OpenVINO and OpenVINO Tokenizers release version perform:\n"
            )
            if is_nightly:
                message += "pip uninstall -y openvino-nightly && "
            message += "pip install --force-reinstall openvino openvino-tokenizers\n"
            if is_nightly:
                message += (
                    "openvino-nightly package will be deprecated in the future - use pre-release drops instead. "
                )
            message += "To update both OpenVINO and OpenVINO Tokenizers to the latest pre-release version perform:\n"
            if is_nightly:
                message += "pip uninstall -y openvino-nightly && "
            message += (
                "pip install --pre -U openvino openvino-tokenizers "
                "--extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly"
            )
        log.warning(message)
        return False

    return True
