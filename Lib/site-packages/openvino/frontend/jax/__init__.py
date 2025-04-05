# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Package: openvino
Low level wrappers for the FrontEnd C++ API.
"""

# flake8: noqa

try:
    from openvino.frontend.jax.py_jax_frontend import _FrontEndJaxDecoder as Decoder
except ImportError as err:
    raise ImportError("OpenVINO JAX frontend is not available, please make sure the frontend is built."
                      "{}".format(err))