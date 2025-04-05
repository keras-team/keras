# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log


def get_jax_decoder(model, args):
    try:
        from openvino.frontend.jax.jaxpr_decoder import JaxprPythonDecoder
    except Exception as e:
        log.error("JAX frontend loading failed")
        raise e
    
    if not isinstance(model, JaxprPythonDecoder):
        decoder = JaxprPythonDecoder(model)
    else:
        decoder = model
    
    args['input_model'] = decoder
