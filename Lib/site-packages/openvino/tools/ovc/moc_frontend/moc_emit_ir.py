# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse

from openvino import Model  # pylint: disable=no-name-in-module,import-error
from openvino.tools.ovc.moc_frontend.preprocessing import apply_preprocessing


def moc_emit_ir(ngraph_function: Model, argv: argparse.Namespace):
    from openvino._offline_transformations import compress_quantize_weights_transformation, \
        apply_moc_transformations  # pylint: disable=no-name-in-module,import-error
    from openvino.tools.ovc.moc_frontend.offline_transformations import apply_moc_legacy_transformations, \
        apply_fused_names_cleanup

    # Apply preprocessing (mean/scale/reverse_channels/convert_layout/etc)
    apply_preprocessing(ov_function=ngraph_function, argv=argv)

    # Apply transformations
    apply_moc_transformations(ngraph_function, cf=False, smart_reshape=True)
    compress_quantize_weights_transformation(ngraph_function)

    if argv.framework == "onnx":  # TODO: Consider removing
        # set OldApi map in IR to be executed via OV API 1.x and for parity with legacy MO
        params_with_custom_types = [] if argv.placeholder_data_types is None \
            else list(argv.placeholder_data_types.keys())
        apply_moc_legacy_transformations(ngraph_function, params_with_custom_types)

    apply_fused_names_cleanup(ngraph_function)

    del argv.feManager
    return ngraph_function
