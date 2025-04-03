# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# flake8: noqa
# mypy: ignore-errors

import jax.core
from openvino.frontend.jax.py_jax_frontend import _FrontEndJaxDecoder as Decoder
from openvino import PartialShape, Type as OVType, OVAny
from openvino.frontend.jax.utils import jax_array_to_ov_const, get_ov_type_for_value, \
    ivalue_to_constant, param_to_constants

import jax
import numpy as np

from typing import List
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

class JaxprPythonDecoder (Decoder):
    '''
    The jaxpr decoder uses Jaxpr to get graph information from a jax module.
    It takes use of the following parts.
    
    - `ClosedJaxpr`: the jaxpr object that contains the jaxpr and literals.
        - `Jaxpr`: the jaxpr object that contains the invars, outvars, and eqns.
            - `JaxEqns`: A list of jaxpr equations, which contains the information of the operation.
                - `Primitive`: the operation that is used in the equation.
                - `invars`: the input variables of the equation.
                    - `aval`: the abstract value.
                - `outvars`: the output variables of the equation.
                    - `aval`: the abstract value.
                - `params`: the named params of this equation.
            - `invars`: the inputs of the model (traced graph).
                - `aval`: the abstract value.
            - `outvars`: the outputs of the model (traced graph).
                - `aval`: the abstract value.
            - `constvars`: the constant variables used in this model.
                - `aval`: the abstract value.
        - `Literal`: the literal object that contains the value of the constants.
    '''
    
    def __init__(self, jaxpr, name=None, literals=None):
        '''
        Inputs: 
            - jaxpr: for users, `ClosedJaxpr` is expected here. See https://github.com/google/jax/blob/jaxlib-v0.4.29/jax/_src/core.py#L197
            - name: the name for the model.
            - literals: the literals (constants) that are used in the model.
        '''
        Decoder.__init__(self)
        
        if isinstance(jaxpr, (jax.core.JaxprEqn, jax.core.Jaxpr)):
            self.jaxpr = jaxpr
        elif isinstance(jaxpr, jax.core.ClosedJaxpr):
            # Take the `Jaxpr` from `ClosedJaxpr`, see https://github.com/google/jax/blob/jaxlib-v0.4.29/jax/_src/core.py#L85
            self.jaxpr = jaxpr.jaxpr
            # Literal should be a `Jax.core.Var`, see https://github.com/google/jax/blob/jaxlib-v0.4.29/jax/_src/core.py#L85
            self.literals = jaxpr.literals
        else:
            raise ValueError(f"Unexpected type of jaxpr: {type(jaxpr)}")
        self.name = name
        if self.name is None:
            self.name = "jax_module"
        if literals is not None:
            self.literals = literals
            
        self.params = {}
        if hasattr(self.jaxpr, 'params') and isinstance(self.jaxpr.params, dict):
            for k in self.jaxpr.params.keys():
                converted = self.convert_param_to_constant_node(self.jaxpr, k)
                if converted is not None:
                    self.params.update(converted)
                
        # TODO: this implementation may lead to memory increasing. Any better solution?
        self.m_decoders = []
        
    def inputs(self) -> List[int]:
        if isinstance(self.jaxpr, jax.core.JaxprEqn):
            idx = 0
            res = []
            for inp in self.jaxpr.invars:
                if isinstance(inp, jax.core.Literal):
                    res.append(self.literals[idx].output(0))
                    idx += 1
                else:
                    res.append(id(inp))
            return res
        else:
            return [id(v) for v in self.jaxpr.invars]
    
    def input(self, idx: int) -> int:
        return id(self.jaxpr.invars[idx])
    
    def get_input_shape(self, index):
        return PartialShape(self.jaxpr.invars[index].aval.shape)
    
    def get_input_signature_name(self, index) -> str:
        return "jaxpr_invar_" + str(index)
    
    def get_input_type(self, index) -> OVType:
        return get_ov_type_for_value(self.jaxpr.invars[index])
        
    def get_named_param(self, name):
        '''
        Get the object id of the named parameter by the name.
        '''
        return self.params[name].output(0)
    
    def get_named_param_as_constant(self, name):
        '''
        The named parameter in JAX is a python object but we want to use its value in cpp.
        Therefore this API is used to get the named parameter as a constant, which can be used
        to extract the value of it in cpp-level.
        '''
        return self.params[name].as_constant()
    
    def get_param_names(self):
        '''
        In JAX, the named parameters may exist in `params` attribute of `JaxEqn`.
        For example, the `jax.lax.cat` operation has a named parameter `dim`, 
        which is used to indicate the dimension to concatenate the tensors.
        
        Here we return the names of all the named params that appear in the model for the current `JaxEqn`.
        '''
        return list(self.params.keys())
    
    def get_output_type(self, index) -> OVType:
        return get_ov_type_for_value(self.jaxpr.outvars[index])
        
    def get_output_name(self, index) -> str:
        return "jaxpr_outvar_" + str(index)
    
    def get_output_shape(self, index):
        return PartialShape(self.jaxpr.outvars[index].aval.shape)
    
    def visit_subgraph(self, node_visitor) -> None:
        if isinstance(self.jaxpr, jax.core.JaxprEqn):
            return
        for _, decoder in self.params.items():
            self.m_decoders.append(decoder)
            node_visitor(decoder)
        for idx, node in enumerate(self.jaxpr.constvars):
            decoder = self.convert_literal_to_constant_node(
                literal=self.literals[idx], 
                name=self.name + "/" + f"const({id(node)})", 
                output_id=id(node)
            )
            self.m_decoders.append(decoder)
            node_visitor(decoder)
        # Visit every `JaxEqn` in the jaxpr, see https://github.com/google/jax/blob/jaxlib-v0.4.29/jax/_src/core.py#L285
        for node in self.jaxpr.eqns:
            literal_decoders = []
            for inp in node.invars:
                if isinstance(inp, jax.core.Literal):
                    literal_decoder = self.convert_literal_to_constant_node(inp)
                    literal_decoders.append(literal_decoder)
                    node_visitor(literal_decoder)
            decoder = JaxprPythonDecoder(node, name=self.name + "/" + node.primitive.name, literals=literal_decoders)
            self.m_decoders.append(decoder)
            node_visitor(decoder)
            
    def get_op_type(self) -> str:
        if isinstance(self.jaxpr, jax.core.JaxprEqn):
            return self.jaxpr.primitive.name
        else:
            return "root"
    
    def outputs(self) -> List[int]:
        return [id(v) for v in self.jaxpr.outvars]
    
    def output(self, idx: int) -> int:
        return id(self.jaxpr.outvars[idx])
    
    def num_inputs(self) -> int:
        return len(self.jaxpr.invars)
    
    def num_outputs(self) -> int:
        return len(self.jaxpr.outvars)
    
    def as_constant(self):
        if self.get_op_type() == 'constant':
            value = self.literals
            # TODO: dig out how to share the memory.
            # Currently, using shared_memory will raise `ValueError: array is not writeable``
            ov_const = jax_array_to_ov_const(value, shared_memory=False)
            return ov_const.outputs()
        else:
            raise ValueError("This is not a constant node so it cannot be converted to a constant.")
        
    @staticmethod
    def convert_param_to_constant_node(jaxpr, param) -> dict:
        assert hasattr(jaxpr, 'params'), "The jaxpr does not have params."
        if hasattr(jaxpr, 'primitive'):
            param_map = param_to_constants(jaxpr.primitive.name, param, jaxpr, shared_memory=False)
            res = {}
            for name, constant in param_map.items():
                if constant is not None:
                    res[name] = _JaxprPythonConstantDecoder(constant=constant)
        else:
            constant = ivalue_to_constant(jaxpr.params[param], shared_memory=False)
            res = {param: _JaxprPythonConstantDecoder(constant=constant)} if constant is not None else {}
        return res
    
    @staticmethod
    def convert_literal_to_constant_node(literal, name=None, output_id=None):
        if isinstance(literal, jax.core.Literal):
            constant = ivalue_to_constant(literal.val, shared_memory=False)
        elif isinstance(literal, (jax.Array, np.ndarray)):
            constant = ivalue_to_constant(literal, shared_memory=False)
        else:
            raise TypeError( f"The input should be a literal or jax array, but got {type(literal)}.")
        return _JaxprPythonConstantDecoder(constant=constant, name=name, output_id=output_id)
        
class _JaxprPythonConstantDecoder (Decoder):
    def __init__(self, name=None, constant=None, output_id=None):
        '''
        A decoder specially for constants and named parameters.
        
        Inputs:
            - name: the name for the model.
            - literals: the literals (constants) that are used in the model.
            - output_id: the id specified for this decoder's output. If none, use `id(self.constant)`.
        '''
        Decoder.__init__(self)
        
        self.name = name
        self.constant = constant
        self.output_id = id(self.constant) if output_id is None else output_id
        
    def inputs(self) -> List[int]:
        return []
    
    def input(self, idx: int) -> int:
        raise ValueError("This is a constant node so it does not have input.")
    
    def get_input_shape(self, index):
        raise ValueError("This is a constant node so it does not have input shape.")
    
    def get_input_signature_name(self, index) -> str:
        raise ValueError("This is a constant node so it does not have input signature name.")
    
    def get_input_type(self, index) -> OVType:
        raise ValueError("This is a constant node so it does not have input type.")
        
    def get_named_param(self, name):
        raise ValueError("This is a constant node so it does not have named param.")
    
    def get_named_param_as_constant(self, name):
        raise ValueError("This is a constant node so it does not have named param.")
    
    def get_param_names(self):
        '''
        In JAX, the named parameters may exist in `params` attribute of `JaxEqn`.
        For example, the `jax.lax.cat` operation has a named parameter `dim`, 
        which is used to indicate the dimension to concatenate the tensors.
        
        However, `_JaxprPythonConstantDecoder` is already a named param or a constant.
        So it will never have a named param.
        '''
        return []
    
    def get_output_type(self, index) -> OVType:
        assert len(self.constant) == 1
        return OVAny(self.constant[0].element_type)
        
    def get_output_name(self, index) -> str:
        return "jaxpr_outvar_" + str(index)
    
    def get_output_shape(self, index):
        assert len(self.constant) == 1
        return PartialShape(self.constant[0].shape)
    
    def visit_subgraph(self, node_visitor) -> None:
        return
            
    def get_op_type(self) -> str:
        return "constant"
    
    def outputs(self) -> List[int]:
        return [self.output_id]
    
    def output(self, idx: int) -> int:
        return self.output_id
    
    def num_inputs(self) -> int:
        return 0
    
    def num_outputs(self) -> int:
        return 1
    
    def as_constant(self):
        return self.constant