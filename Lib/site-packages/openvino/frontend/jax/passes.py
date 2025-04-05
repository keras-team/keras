# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# flake8: noqa
# mypy: ignore-errors

from enum import Enum
from jax.lax import ConvDimensionNumbers

def enum_values_pass(value):
    if isinstance(value, Enum):
        return value.value
    return value


def conv_dimension_numbers_pass(value):
    if isinstance(value, ConvDimensionNumbers):
        return [
            list(value.lhs_spec),
            list(value.rhs_spec),
            list(value.out_spec)
        ]
    return value


def filter_element(value):
    passes = [enum_values_pass]
    for pass_ in passes:
        value = pass_(value)
    return value


def filter_ivalue(value):
    passes = [conv_dimension_numbers_pass]
    for pass_ in passes:
        value = pass_(value)
    return value


def dot_general_param_pass(param_name: str, jax_eqn):
    param = jax_eqn.params[param_name]
    res = {}
    if param_name == 'dimension_numbers':
        contract_dimensions = param[0]
        assert len(contract_dimensions) == 2
        res['contract_dimensions'] = [list(contract_dimensions[0]), list(contract_dimensions[1])]
        
        batch_dimensions = param[1]
        assert len(batch_dimensions) == 2
        lhs_length = len(batch_dimensions[0])
        rhs_length = len(batch_dimensions[1])
        assert lhs_length == rhs_length
        if lhs_length > 0:
            res['batch_dimensions'] = [list(batch_dimensions[0]), list(batch_dimensions[1])]
    return res

# mapping from primitive to pass 
param_passes = {
    'dot_general': dot_general_param_pass,
}

def filter_param(primitive: str, param_name: str, jax_eqn):
    if primitive in param_passes:
        return param_passes[primitive](param_name, jax_eqn)
    return {param_name: jax_eqn.params[param_name]}
