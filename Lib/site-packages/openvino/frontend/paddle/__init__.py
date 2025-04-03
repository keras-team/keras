# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Package: openvino
Low level wrappers for the FrontEnd C++ API.
"""

# flake8: noqa

try:
    from openvino.frontend.paddle.py_paddle_frontend import ConversionExtensionPaddle as ConversionExtension
    from openvino.frontend.paddle.py_paddle_frontend import OpExtensionPaddle as OpExtension
except ImportError as err:
    raise ImportError("OpenVINO Paddle frontend is not available, please make sure the frontend is built." "{}".format(err))
