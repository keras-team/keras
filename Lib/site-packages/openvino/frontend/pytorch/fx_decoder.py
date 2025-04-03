# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# flake8: noqa
# mypy: ignore-errors

import logging
import inspect
import torch

from openvino.frontend.pytorch.py_pytorch_frontend import _FrontEndPytorchDecoder as Decoder
from openvino.frontend.pytorch.py_pytorch_frontend import _Type as DecoderType
from openvino import PartialShape, Type as OVType, OVAny, Shape
from openvino.frontend.pytorch.utils import (
    make_constant, fetch_attr, pt_to_ov_type_map, torch_tensor_to_ov_const)

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class BaseFXDecoder (Decoder):
    """
    BaseFXDecoder is a class that extends the Decoder class to handle decoding
    operations for FX graphs in PyTorch. It provides a common interface for all
    FX decoders.
    """

    def __init__(self, mark_node_callback=None) -> None:
        Decoder.__init__(self)
        self.mark_node_callback = mark_node_callback
        # We store every decoder created by this decoder so that
        # all them are not deleted until the first decoder is deleted
        self.m_decoders = []
        self._inputs = []
        self._outputs = []

    @staticmethod
    def unpack_containers(arg):
        if isinstance(arg, (tuple, list)):
            res = []
            for e in arg:
                res.extend(BaseFXDecoder.unpack_containers(e))
            return res
        elif isinstance(arg, dict):
            res = []
            for k, e in arg.items():
                unpacked = BaseFXDecoder.unpack_containers(e)
                if len(unpacked) == 1:
                    unpacked[0] = (k, unpacked[0][1])
                res.extend(unpacked)
            return res
        else:
            return [("", arg)]

    @staticmethod
    def arg_to_constant(arg):
        if isinstance(arg, list):
            if len(arg) > 0:
                return make_constant(pt_to_ov_type_map[type(
                    arg[0]).__name__], Shape([len(arg)]), arg)
            else:
                # TODO: which type should we use if list is empty? Need a signaling value here
                return make_constant(OVType.i32, Shape([0]), [])
        elif isinstance(arg, bool):
            return make_constant(OVType.boolean, Shape([]), [arg])
        elif isinstance(arg, int):
            return make_constant(OVType.i64, Shape([]), [arg])
        elif isinstance(arg, float):
            return make_constant(OVType.f32, Shape([]), [arg])
        elif isinstance(arg, str):
            u8_tensor = torch.frombuffer(str.encode(arg), dtype=torch.uint8)
            return torch_tensor_to_ov_const(u8_tensor, shared_memory=True)
        return None

    @staticmethod
    def get_type_for_value(value):
        if issubclass(type(value), torch.fx.Node):
            if ('tensor_meta' in value.meta.keys()):
                if value.meta['tensor_meta'] and isinstance(value.meta['tensor_meta'], torch.Tensor):
                    pt_type = value.meta['tensor_meta'].dtype
                    if str(pt_type) in pt_to_ov_type_map:
                        ov_type = pt_to_ov_type_map[str(pt_type)]
                        return OVAny(ov_type)
            return OVAny(OVType.dynamic)
        elif isinstance(value, int):
            return OVAny(DecoderType.PyScalar(OVAny(OVType.i64)))
        elif isinstance(value, float):
            return OVAny(DecoderType.PyScalar(OVAny(OVType.f32)))
        elif isinstance(value, bool):
            return OVAny(DecoderType.PyScalar(OVAny(OVType.boolean)))
        elif isinstance(value, list):
            if len(value) > 0:
                return OVAny(DecoderType.List(BaseFXDecoder.get_type_for_value(value[0])))
            else:
                return OVAny(DecoderType.List(OVAny(OVType.i32)))
        return OVAny(OVType.dynamic)

    def inputs(self):
        # Consider 0 a special case which may mean the input is inlined, but not guaranteed
        return [x if not isinstance(x, InlinedInput) else 0 for x in self._inputs]

    def input(self, index):
        return self.inputs()[index]

    def output(self, index):
        return self.outputs()[index]

    def get_input_debug_name(self, index):
        return "input" + str(index)

    def is_input_inlined(self, index):
        return isinstance(self._inputs[index], InlinedInput)

    def get_inlined_input_decoder(self, index):
        target = self._inputs[index]
        assert isinstance(target, InlinedInput), "Requested non-inlined input"
        in_decoder = InlinedInputDecoder(
            target, self._nodes, self.mark_node_callback)
        self.m_decoders.append(in_decoder)
        return in_decoder

    def get_input_shape(self, index):
        return PartialShape.dynamic()

    def get_input_type(self, index):
        return OVAny(OVType.dynamic)

    def get_output_type(self, index):
        return OVAny(OVType.dynamic)

    def input_is_none(self, index):
        if index < len(self._inputs) and isinstance(self._inputs[index], InlinedInput):
            return self._inputs[index].data is None
        return False

    def decoder_type_name(self) -> str:
        return "fx"

    def get_schema(self):
        return 'NONE'

    def mark_node(self, node):
        if self.mark_node_callback is not None:
            self.mark_node_callback(self, node)
        return node

    def get_subgraphs(self):
        return []

    def get_subgraph_size(self):
        return len(self.get_subgraphs())

    def as_string(self):
        return None

    def may_produce_alias(self, in_index: int, out_index: int) -> bool:
        return False

    def get_rt_info(self):
        rt_info = {}
        return rt_info


class TorchFXPythonDecoder (BaseFXDecoder):
    """
    Decoder for PyTorch FX GraphModule and Node objects to OpenVINO IR.
    """

    def __init__(self, pt_module, fx_gm=None, nodes=None,
                 mark_node_callback=None, input_shapes=[], input_types=[], dynamic_shapes=False):
        super().__init__(mark_node_callback)
        self.pt_module = pt_module
        self.fx_gm = fx_gm if fx_gm is not None else pt_module
        self.input_types = [OVAny(pt_to_ov_type_map[str(t)])
                            for t in input_types]
        self.input_shapes = input_shapes

        self._input_signature = []
        self._example_input = None

        if issubclass(type(pt_module), torch.fx.graph_module.GraphModule):
            self._input_is_list = None
            self._nodes = list(pt_module.graph.nodes)
            found_types = []
            found_shapes = []
            for i, value in enumerate(self._nodes):
                if value.op == 'placeholder':
                    self._inputs.append(i)
                    self._input_signature.append(value.name)
                    if hasattr(value, "meta") and ('tensor_meta' in value.meta.keys()) and value.meta['tensor_meta']:
                        found_shapes.append(value.meta['tensor_meta'].shape)
                        found_types.append(
                            OVAny(pt_to_ov_type_map[str(value.meta['tensor_meta'].dtype)]))
                    else:
                        found_shapes.append(None)
                        found_types.append(None)
                elif value.op == 'output':
                    # Instead of putting output index, refer to its target
                    uargs = self.unpack_containers(value.args)
                    self._outputs = [(arg[0], self._nodes.index(arg[1]))
                                     for arg in uargs if arg[1] is not None]
            for idx, shape in enumerate(found_shapes):
                if shape is not None:
                    new_shape = []
                    for dim in shape:
                        if (dynamic_shapes or type(dim).__name__ == "SymInt"):
                            new_shape.append(-1)
                        else:
                            new_shape.append(dim)
                    found_shapes[idx] = torch.Size(new_shape)

            if not input_shapes or len(input_shapes) == 0:
                self.input_shapes = found_shapes
            if not input_types or len(input_types) == 0:
                self.input_types = found_types

            if hasattr(pt_module, "forward"):
                input_params = inspect.signature(pt_module.forward).parameters
                self._input_signature = list(input_params)

        elif issubclass(type(pt_module), torch.fx.Node):
            self._nodes = nodes  # passed from outer context

            # FIXME: Quadratic complexity nodes*nodes considering the outer loop over all nodes
            self._outputs = [("", self._nodes.index(pt_module))]

            self.input_types = []
            for arg in pt_module.args:
                if isinstance(arg, torch.fx.Node):
                    self._inputs.append(self._nodes.index(arg))
                else:
                    # Not a node, consider it inlined
                    self._inputs.append(InlinedInput(arg))
                self.input_types.append(
                    BaseFXDecoder.get_type_for_value(arg))

    def get_input_signature_name(self, index: int) -> str:
        if self._input_signature is not None and index < len(self._input_signature):
            return self._input_signature[index]
        return self.get_input_debug_name(index)

    def get_input_shape(self, index):
        if index < len(self.input_shapes) and self.input_shapes[index] is not None:
            return PartialShape(self.input_shapes[index])
        input = self._raw_input(index)
        return self.get_shape_for_value(input)

    def get_input_strides(self, index: int) -> list:
        raw_input = self._raw_input(index)
        if isinstance(raw_input, torch.fx.node.Node) and hasattr(raw_input, "meta"):
            meta = raw_input.meta
            if "tensor_meta" in meta and hasattr(meta["tensor_meta"], "stride"):
                strides = list(meta["tensor_meta"].stride)
                if strides:
                    return strides
        return []

    def get_input_type(self, index):
        if index < len(self.input_types) and self.input_types[index] is not None:
            return self.input_types[index]
        input = self._raw_input(index)
        return self.get_type_for_value(input)

    def get_output_debug_name(self, index):
        if self._outputs is not None and index < len(self._outputs) and self._outputs[index][0]:
            return self._outputs[index][0]
        name = getattr(self.pt_module, "name", "output")
        return name + ":" + str(index)

    def get_output_shape(self, index):
        output = self._raw_output(index)
        return self.get_shape_for_value(output)

    def get_output_type(self, index):
        output = self._raw_output(index)
        return self.get_type_for_value(output)

    def get_shape_for_value(self, value):
        if value and hasattr(value, "meta") and ('tensor_meta' in value.meta.keys()):
            if value.meta['tensor_meta']:
                return PartialShape(len(value.meta['tensor_meta'].shape) * [-1])
        return PartialShape.dynamic()

    def get_attribute(self, name):
        if name in self.pt_module.kwargs:
            attr = self.pt_module.kwargs[name]
            if isinstance(attr, torch.dtype):
                return OVAny(pt_to_ov_type_map[str(attr)])
            if isinstance(attr, torch.device):
                return OVAny(attr.type)
            if isinstance(attr, str):
                return OVAny(attr)
            # Numeric attrs convert to Constant
            constant = self.arg_to_constant(attr)
            if constant is not None:
                return OVAny(constant.output(0))
            # so that has_attribute return True if attribute exist
            return OVAny(DecoderType.PyNone())
        return OVAny(None)

    def get_named_input(self, name):
        """
        Returns id of kwargs input. Such input can be Node or a constant value,
        this function is only used for to return node index. If the input is
        constant, get_attribute should be used.
        """
        if name in self.pt_module.kwargs:
            arg = self.pt_module.kwargs[name]
            if isinstance(arg, torch.fx.Node):
                return self._nodes.index(arg)
        raise RuntimeError("This input is not a Node")

    def visit_subgraph(self, node_visitor):
        # make sure topological order is satisfied
        for node in self._nodes:
            if node.op == 'placeholder' or node.op == 'output':
                continue  # skipping non-operational nodes
            if node.op == 'call_function' and str(node.target) in ["aten._assert_async.msg"]:
                continue
            decoder = TorchFXPythonDecoder(
                node, self.fx_gm, self._nodes, mark_node_callback=self.mark_node_callback)
            self.m_decoders.append(decoder)
            node_visitor(decoder)

    def get_subgraph_decoder(self, index):
        decoder = TorchFXPythonDecoder(self.get_subgraphs()[index],
                                       self.fx_gm,
                                       mark_node_callback=self.mark_node_callback)
        self.m_decoders.append(decoder)
        return decoder

    def get_op_type(self):
        if self.pt_module.op == 'call_function':
            return str(self.pt_module.target)
        elif self.pt_module.op == 'get_attr':
            return 'get_attr'  # FIXME should be aligned with get_attr from TS implementation
        else:
            return 'UNKNOWN_TYPE_' + str(self.pt_module.op)

    def outputs(self):
        return [o[1] for o in self._outputs]

    def _raw_outputs(self):
        return [self._nodes[x[1]] for x in self._outputs]

    def _raw_output(self, index):
        return self._raw_outputs()[index]

    def _raw_inputs(self):
        return [self._nodes[x] if not isinstance(x, InlinedInput) and x < len(self._nodes) else x.data for x in self._inputs]

    def _raw_input(self, index):
        return self._raw_inputs()[index]

    def num_of_outputs(self):
        return len(self.outputs())

    def output_list_size(self):
        max_out_id = -1
        for user in self.pt_module.users:
            if "<built-in function getitem>" == str(user.target) and max_out_id < user.args[1]:
                max_out_id = user.args[1]
        return max_out_id + 1

    def mark_node(self, node):
        name = self.get_op_type()
        if "FrameworkNode" not in node.get_type_name():
            name += "/" + node.get_type_name()
        node.set_friendly_name(self.pt_module.name + "/" + name)
        super().mark_node(node)
        return node

    def as_constant(self):
        assert self.pt_module.op == 'get_attr', "Only get_attr is supported"
        # Extract Constant from FX module field
        ret = fetch_attr(self.fx_gm, self.pt_module.target)
        ov_const = torch_tensor_to_ov_const(ret, shared_memory=True)
        return ov_const.outputs()

    def input_is_none(self, index):
        if index >= len(self._inputs) or (isinstance(self._inputs[index], InlinedInput) and self._inputs[index].data is None):
            return True
        else:
            r_input = self._raw_input(index)
            return str(type(r_input)) in ['torch.NoneType', 'NoneType']

    def debug(self):
        self.pt_module.print()


class InlinedInput:
    """
    Represents an inlined input. This is a special case
    where the input is not a node, but a constant value.
    """

    def __init__(self, data) -> None:
        self.data = data


class InlinedInputDecoder (BaseFXDecoder):
    """
    Decoder for inlined inputs in PyTorch FX graphs.
    """

    def __init__(self, inlined_input: InlinedInput, nodes=None, mark_node_callback=None) -> None:
        super().__init__(mark_node_callback)
        self.inlined_input = inlined_input
        self._nodes = nodes
        self.is_const = not (isinstance(inlined_input.data, (list, tuple)) and any(
            isinstance(a, torch.fx.Node) for a in inlined_input.data))
        if not self.is_const:
            self._inputs = [nodes.index(x) if isinstance(
                x, torch.fx.Node) else InlinedInput(x) for x in inlined_input.data]

    def get_op_type(self):
        # return specific type for inlined inputs
        if not self.is_const:
            return "prim::ListConstruct"
        return "inlined.constant.default"

    def outputs(self):
        return [0]

    def num_of_outputs(self):
        return 1

    def get_input_shape(self, index):
        return PartialShape.dynamic()

    def get_input_type(self, index):
        return OVAny(OVType.dynamic)

    def get_output_type(self, index):
        return OVAny(OVType.dynamic)

    def input_is_none(self, index):
        if index < len(self._inputs) and isinstance(self._inputs[index], InlinedInput):
            return self._inputs[index].data is None
        return False

    def as_constant(self):
        arg = self.inlined_input.data
        constant = BaseFXDecoder.arg_to_constant(arg)
        if constant is not None:
            return constant.outputs()
        return []

    def mark_node(self, node):
        name = self.get_op_type()
        if "FrameworkNode" not in node.get_type_name():
            name += "/" + node.get_type_name()
        node.set_friendly_name(name)
        super().mark_node(node)
        return node
