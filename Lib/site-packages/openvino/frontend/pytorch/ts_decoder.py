# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# flake8: noqa
# mypy: ignore-errors

import inspect
import logging
import typing
import torch

from openvino.frontend.pytorch.py_pytorch_frontend import _FrontEndPytorchDecoder as Decoder
from openvino.frontend.pytorch.py_pytorch_frontend import _Type as DecoderType
from openvino import op, PartialShape, Type as OVType, OVAny
from openvino.frontend.pytorch.utils import (
    ivalue_to_constant,
    get_value_from_getattr,
    pt_to_ov_type_map,
    prepare_example_inputs_and_model,
    convert_quantized_tensor,
    graph_has_ops,
    patch_none_example,
)
from openvino import opset11 as ops
from openvino.frontend.pytorch import quantized, patch_model
from openvino.frontend.pytorch.module_extension import ModuleExtension

log = logging.getLogger(__name__)


class TorchScriptPythonDecoder(Decoder):
    def __init__(
        self,
        pt_module,
        graph_element=None,
        example_input=None,
        alias_db=None,
        shared_memory=True,
        skip_freeze=False,
        constant_cache=None,
        module_extensions=None,
        trace_kwargs=None,
    ):
        super().__init__()
        # We store every decoder created by this decoder so that all them are not deleted until the first decoder is deleted
        self.m_decoders = []
        self._input_signature = None
        self._shared_memory = shared_memory
        self._input_is_list = False
        self.constant_cache = constant_cache if constant_cache is not None else dict()
        self.module_extensions = module_extensions
        self.config = None
        self.out_debug_name_overwrites = {}
        if graph_element is None:
            if hasattr(pt_module, "config"):
                if isinstance(pt_module.config, dict):
                    self.config = pt_module.config
                elif hasattr(pt_module.config, "to_dict"):
                    self.config = pt_module.config.to_dict()
            try:
                pt_module = self._get_scripted_model(
                    pt_module, example_input, skip_freeze, trace_kwargs)
            except Exception as e:
                if example_input is not None:
                    msg = "tracing"
                    help_msg = "Please check correctness of provided 'example_input'. " \
                        "Sometimes models can be converted in scripted mode, please try running " \
                        "conversion without 'example_input'.\n"
                else:
                    msg = "scripting"
                    help_msg = "Tracing sometimes provide better results, " \
                        "please provide valid 'example_input' argument.\n"
                raise RuntimeError(
                    f"Couldn't get TorchScript module by {msg}.\n{help_msg} "
                    "You can also provide TorchScript module that you obtained"
                    " yourself, please refer to PyTorch documentation: "
                    "https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html."
                ) from e
            self.graph_element = pt_module.inlined_graph
            self.alias_db = self.graph_element.alias_db()
        else:
            self.graph_element = graph_element
            self.alias_db = alias_db
        self.pt_module = pt_module
        self.raw_inputs = list(self.graph_element.inputs())
        self.raw_outputs = list(self.graph_element.outputs())
        if self._input_signature is not None:
            if "self" in self.raw_inputs[0].debugName():
                self._input_signature.insert(0, "self")
            if 0 < len(self._input_signature) < len(self.raw_inputs):
                # last input is args input, we need to multiply that name by number of extra inputs
                self._input_signature = self._input_signature[:-1]
                n = len(self._input_signature)
                for i in range(len(self.raw_inputs) - n):
                    self._input_signature.append(
                        self.raw_inputs[i + n].debugName())

        if isinstance(self.graph_element, torch.Graph):
            self._transform_tensor_list_constants_to_listconstruct(
                self.graph_element)
            self._transform_optional_constants(self.graph_element)
            log.debug("Inlined graph:\n%s", self.graph_element)

    @staticmethod
    def _get_preserved_attributes(model) -> list:
        preserved_attributes = []
        for name, module in model.named_modules():
            compressed_types = [torch.int8, torch.uint8,
                                torch.float16, torch.bfloat16]
            if hasattr(module, "weight") and getattr(module.weight, "dtype", None) in compressed_types:
                preserved_attributes.append(name)
        return preserved_attributes

    def _get_scripted_model(self, pt_module, example_inputs=None, skip_freeze=False, trace_kwargs=None):
        freeze_by_default = False
        if isinstance(pt_module, torch.nn.Module):
            pt_module.eval()
        input_signature = None
        input_parameters = None
        if isinstance(pt_module, torch.nn.Module) and not isinstance(
            pt_module, (torch.jit._trace.TopLevelTracedModule,
                        torch.jit._script.RecursiveScriptModule)
        ):
            # input params is dictionary contains input names and their signature values (type hints and default values if any)
            input_params = inspect.signature(pt_module.forward if hasattr(
                pt_module, "forward") else pt_module.__call__).parameters
            input_signature = list(input_params)

            if example_inputs is None:
                if self.module_extensions:
                    raise RuntimeError(
                        "ModuleExtension is not supported for scripting. "
                        "Please provide valid example_input argument to run tracing.")
                scripted = torch.jit.script(pt_module)
                freeze_by_default = True
            else:
                pt_module, example_inputs = patch_none_example(pt_module, example_inputs)
                input_parameters, input_signature, pt_module, self._input_is_list = prepare_example_inputs_and_model(
                    example_inputs, input_params, pt_module)

                # name of attribute in a patched module where the original forward method is kept
                orig_forward_name = "_openvino_module_extension_patch_orig_forward"
                if self.module_extensions:
                    patch_model.patch_model(
                        pt_module, self.module_extensions, orig_forward_name)

                patched = False
                if quantized.detect_quantized_model(pt_module) is not None:
                    try:
                        quantized.patch_quantized(pt_module)
                        patched = True
                    except Exception as error:
                        log.warning(
                            "Failed patching of AutoGPTQ model. Error message:\n"
                            "Tracing of the model will likely be unsuccessful or incorrect",
                            exc_info=error)
                        quantized.unpatch_quantized(pt_module)
                        patched = False

                if trace_kwargs is None:
                    trace_kwargs = {}
                try:
                    scripted = torch.jit.trace(
                        pt_module, **input_parameters, strict=False, **trace_kwargs)
                finally:
                    if patched:
                        quantized.unpatch_quantized(pt_module)

            have_to_freeze_ops = ["prim::Uninitialized",
                                  "prim::unchecked_cast", "aten::append"]
            if not freeze_by_default and graph_has_ops(scripted.inlined_graph, have_to_freeze_ops):
                # freeze models with unsupported ops
                freeze_by_default = True
            quantized_hint_ops = ["quantized", "aten::as_strided"]
            if freeze_by_default and graph_has_ops(scripted.inlined_graph, quantized_hint_ops):
                # do not freeze quantized models and can't freeze for aten::as_strided it will result in incorrect inference
                freeze_by_default = False
            if freeze_by_default and not skip_freeze:
                preserved_attrs = self._get_preserved_attributes(scripted)
                f_model = torch.jit.freeze(
                    scripted, preserved_attrs=preserved_attrs)
            else:
                f_model = scripted
            self._example_input = input_parameters["example_inputs"] if input_parameters else None
        else:
            f_model = pt_module
            self._example_input = example_inputs

        self._input_signature = input_signature
        return f_model

    def inputs(self) -> list:
        return [x.unique() for x in self.raw_inputs]

    def get_input(self, index: int):
        return self.inputs()[index]

    def get_input_debug_name(self, index: int) -> str:
        return self._raw_input(index).debugName()

    def get_input_signature_name(self, index: int) -> str:
        if self._input_signature is not None and index < len(self._input_signature):
            return self._input_signature[index]
        return self.get_input_debug_name(index)

    def get_input_shape(self, index: int):
        raw_input = self._raw_input(index)
        return self.get_shape_for_value(raw_input)

    def get_input_strides(self, index: int) -> typing.List[int]:
        raw_input = self._raw_input(index)
        if isinstance(raw_input, torch.Value):
            inp_type = raw_input.type()
            if isinstance(inp_type, torch.TensorType):
                strides = inp_type.strides()
                if strides:
                    return strides
        return []

    def get_input_type(self, index: int):
        raw_input = self._raw_input(index)
        return self.get_type_for_value(raw_input)

    def get_output_debug_name(self, index: int) -> str:
        if index in self.out_debug_name_overwrites:
            return self.out_debug_name_overwrites[index]
        return self._raw_output(index).debugName()

    def get_output_shape(self, index: int):
        output = self._raw_output(index)
        return self.get_shape_for_value(output)

    def get_output_type(self, index: int):
        output = self._raw_output(index)
        return self.get_type_for_value(output)

    def _get_known_type_for_value(self, pt_type):
        """Returns known/unknown types wrapped as OVAny."""
        # Check for simple scalar types first
        if pt_type is None:
            return OVAny(OVType.dynamic)
        # TODO: Don't use str, use native types
        if str(pt_type) in ["int", "float", "bool"]:
            return OVAny(DecoderType.PyScalar(OVAny(pt_to_ov_type_map[str(pt_type)])))
        elif str(pt_type) in pt_to_ov_type_map:
            return OVAny(pt_to_ov_type_map[str(pt_type)])
        elif isinstance(pt_type, torch.TensorType):
            # Tensor type, parse element type
            return OVAny(DecoderType.Tensor(self._get_known_type_for_value(pt_type.dtype())))
        elif isinstance(pt_type, torch.ListType):
            element_type = pt_type.getElementType()
            return OVAny(DecoderType.List(self._get_known_type_for_value(element_type)))
        elif isinstance(pt_type, (torch.StringType, torch.DeviceObjType)):
            return OVAny(DecoderType.Str())
        elif isinstance(pt_type, torch.NoneType):
            return OVAny(DecoderType.PyNone())
        else:
            # Not yet recognized
            return OVAny(OVType.dynamic)

    def get_shape_for_value(self, value: torch.Value):
        if value.isCompleteTensor():
            # We avoid static shapes, they don't generalize on other inputs
            ps = PartialShape([-1] * len(value.type().sizes()))
            return ps
        else:
            # TODO: Recognize types that we can represent as a nested constructs with objects from DecoderType
            # If recognized, return scalar instead of dynamic. Scalar means a single value of that custom type.
            # See get_type_for_value for reference
            pass
        return PartialShape.dynamic()

    def get_type_for_value(self, value: torch.Value):
        full_type = self._get_known_type_for_value(value.type())
        return full_type

    def get_subgraph_size(self) -> int:
        if isinstance(self.graph_element, torch.Node):
            return len(self.get_subgraphs())
        else:
            return 1

    def visit_subgraph(self, node_visitor) -> None:
        # make sure topological order is satisfied
        for node in self.graph_element.nodes():
            decoder = TorchScriptPythonDecoder(
                self.pt_module,
                node,
                alias_db=self.alias_db,
                shared_memory=self._shared_memory,
                constant_cache=self.constant_cache,
                module_extensions=self.module_extensions,
            )
            self.m_decoders.append(decoder)
            node_visitor(decoder)

    def decoder_type_name(self) -> str:
        return "ts"

    def get_subgraphs(self) -> list:
        if self.graph_element.kind() in ["prim::PythonOp", "prim::fork"]:
            if "Subgraph" in self.graph_element.attributeNames():
                assert isinstance(
                    self.graph_element, torch.Node), "Graph element must be of type torch.Node."
                subgraph = getattr(self.graph_element, self.graph_element.kindOf("Subgraph"))("Subgraph")
                torch._C._jit_pass_inline(subgraph)
                return [subgraph]
            else:
                # Attribute "Subgraph" is only available if Graph was created using tracing.
                # TODO Find way to extract subgraph for scripted Graph.
                return []
        return list(self.graph_element.blocks())

    def get_subgraph_decoder(self, index: int):
        module = self.pt_module
        if self.graph_element.kind() == "prim::fork":
            in0 = self.raw_inputs[0]
            if in0.node().kind() == "prim::GetAttr":
                module, _ = get_value_from_getattr(in0.node(), self.pt_module)
        decoder = TorchScriptPythonDecoder(module,
                                           self.get_subgraphs()[index],
                                           alias_db=self.alias_db,
                                           shared_memory=self._shared_memory,
                                           module_extensions=self.module_extensions
                                           )
        self.m_decoders.append(decoder)
        return decoder

    def get_op_type(self) -> str:
        assert isinstance(
            self.graph_element, torch.Node), "Function can be called only when self.graph_element is of type torch.Node"
        if self.graph_element.kind() == "prim::PythonOp" and callable(getattr(self.graph_element, "pyobj", None)):
            pyobj = self.graph_element.pyobj()
            trampoline = getattr(pyobj, "__self__", None)
            target_extension = getattr(trampoline, "target_extension", None)

            if isinstance(target_extension, ModuleExtension):
                target_op = target_extension.target_op
                if callable(target_op):
                    target = target_op(trampoline.original_module)
                elif isinstance(target_op, str):
                    target = target_op
                # TODO: Support target as a callable that will play a role of ConversionExtension for an entire module instead of a single op.
                # Without supporting target as a callable here, ConversionExtension functionality is still possible to implement
                # by combining two extensions: ModuleExtension that use temporary name as a target op and another extension of type ConversionExtension
                # that translates that particular temporary name to custom graph. But providing conversion code as a callable `target` is more convenient.
                return target
        return self.graph_element.kind()

    def get_schema(self) -> str:
        return self.graph_element.schema()

    def outputs(self) -> list:
        return [x.unique() for x in self.raw_outputs]

    def _raw_output(self, index: int):
        return self.raw_outputs[index]

    def _raw_input(self, index: int):
        return self.raw_inputs[index]

    def num_of_outputs(self):
        return len(self.raw_outputs)

    def output(self, index: int):
        return self.outputs()[index]

    def mark_node(self, node):
        name = self.get_op_type()
        if "FrameworkNode" not in node.get_type_name():
            name += "/" + node.get_type_name()
        if self.graph_element.scopeName():
            node.set_friendly_name(
                self.graph_element.scopeName().split("/")[-1] + "/" + name)
        else:
            node.set_friendly_name(name)
        return node

    def _add_name_to_const_and_cache(self, outputs, name):
        if len(outputs) == 1:
            # set name corresponding to state_dict name
            outputs[0].get_node().set_friendly_name(name)
            self.out_debug_name_overwrites[0] = name
        self.constant_cache[name] = outputs

    def try_decode_get_attr(self):
        pt_value, name = get_value_from_getattr(
            self.graph_element, self.pt_module)
        assert pt_value is not None, "Couldn't retrieve value from prim::GetAttr"
        if isinstance(pt_value, torch.ScriptObject):
            # We assume this is __torch__.torch.classes.quantized.Conv2dPackedParamsBase or __torch__.torch.classes.quantized.LinearPackedParamsBase
            # TODO: but can be anything. Figure a better way to distinguish
            weight, bias = pt_value.unpack()
            w_name = name + ".weight"
            if w_name in self.constant_cache:
                res = self.constant_cache[w_name]
            else:
                res = convert_quantized_tensor(weight, self._shared_memory)
                self._add_name_to_const_and_cache(res, w_name)

            if isinstance(bias, torch.Tensor):
                b_name = name + ".bias"
                if b_name in self.constant_cache:
                    res += self.constant_cache[b_name]
                else:
                    b_res = ivalue_to_constant(bias)
                    self._add_name_to_const_and_cache(b_res, b_name)
                    res += b_res
            else:
                res += ops.convert_like(ivalue_to_constant(torch.zeros(1))
                                        [0], res[0]).outputs()
            try:
                # these params exist only for conv params
                stride = pt_value.stride()
                padding = pt_value.padding()
                dilation = pt_value.dilation()
                groups = pt_value.groups()
                res += ivalue_to_constant(stride,
                                          shared_memory=self._shared_memory)
                res += ivalue_to_constant(padding,
                                          shared_memory=self._shared_memory)
                res += ivalue_to_constant(dilation,
                                          shared_memory=self._shared_memory)
                res += ivalue_to_constant(groups,
                                          shared_memory=self._shared_memory)
            except:
                pass
            return res
        elif not isinstance(pt_value, (torch.jit.ScriptModule, torch.jit.TracedModule)):
            # this tensor can be used multiple times in the model, so we have to reuse constants
            if name in self.constant_cache:
                const = self.constant_cache[name]
            else:
                const = ivalue_to_constant(
                    pt_value, shared_memory=self._shared_memory)
                self._add_name_to_const_and_cache(const, name)
            return const
        else:
            return []

    def as_constant(self):
        if not isinstance(self.graph_element, torch.Node):
            return None
        if not self.get_op_type() == "prim::Constant":
            return None
        pt_value = self._raw_output(0)
        pt_type = pt_value.type()
        if isinstance(pt_type, torch.TensorType):
            return ivalue_to_constant(pt_value.toIValue(), shared_memory=self._shared_memory)
        if isinstance(pt_type, torch.ListType):
            return self._as_constant_list(pt_value)
        const = ivalue_to_constant(
            pt_value.toIValue(), shared_memory=self._shared_memory)
        if len(const) > 0:
            # set name corresponding to state_dict name
            const[0].get_node().set_friendly_name(
                self.get_output_debug_name(0))
        return const

    def as_string(self):
        if self.get_op_type() == "prim::Constant":
            pt_value = self._raw_output(0)
            if str(pt_value.type()) in ["torch.StringType", "str"]:
                return pt_value.toIValue()
            elif str(pt_value.type()) == "Device":
                return pt_value.toIValue().type
        elif self.get_op_type() == "prim::device":
            return self._get_device_string()
        return None

    @staticmethod
    def _as_constant_list(pt_value: torch.Value):
        # For now we treat a list as a 1D tensor; it is required by converters to avoid
        # need to massively rewrite them in that part where constant attributes are queried
        pt_element_type = str(pt_value.type().getElementType())
        ivalue = pt_value.toIValue()
        is_known_type = pt_element_type in pt_to_ov_type_map

        if is_known_type:
            ovtype = pt_to_ov_type_map[pt_element_type]
            ovshape = PartialShape([len(ivalue)])
            ov_const = op.Constant(ovtype, ovshape.get_shape(), ivalue)
            return ov_const.outputs()
        return []

    def _get_device_string(self) -> str:
        assert self.graph_element.kind(
        ) == "prim::device", "This function can be called for prim::device node."
        value = self.raw_inputs[0]
        if value.type().isSubtypeOf(torch.TensorType.get()):
            tensor = typing.cast(torch.TensorType, value.type())
            device = tensor.device()
            if device:
                return str(device)
        # Device cannot be statically determined.
        return "cpu"

    def input_is_none(self, index: int) -> bool:
        if index >= len(self.inputs()) or self._raw_input(index) is None:
            return True
        else:
            r_input = self._raw_input(index)
            if str(r_input.type()) in ["torch.NoneType", "NoneType"]:
                return True
            else:
                in_node = r_input.node()
                if in_node.kind() == "prim::GetAttr":
                    pt_value, _ = get_value_from_getattr(
                        in_node, self.pt_module)
                    return pt_value is None
        return False

    def may_produce_alias(self, in_index: int, out_index: int) -> bool:
        if self.get_op_type() in ["aten::conv1d", "aten::conv2d", "aten::conv3d", "aten::_convolution", "aten::matmul", "aten::clone"]:
            # AliasDB::may_contain_alias sometimes return True for tensors produced by convolution or matmul, we have to workaround that
            return False
        try:
            return self.alias_db.may_contain_alias(self._raw_input(in_index), self._raw_output(out_index))
        except:
            # Sometimes pytorch fails to get result with IndexError exception while these indexes exist in node
            return False

    def is_input_inlined(self, index):
        return False

    def get_inlined_input_decoder(self, index):
        return None

    def get_attribute(self, name):
        return OVAny(None)

    def get_named_input(self, name):
        raise RuntimeError("There is no named inputs in TS graph")

    def get_rt_info(self):
        rt_info = {}
        if self.config is not None and "quantization_config" in self.config and "sym" in self.config["quantization_config"]:
            rt_info["symmetric_quantization"] = OVAny(
                self.config["quantization_config"]["sym"])
        return rt_info

    @staticmethod
    def _transform_tensor_list_constants_to_listconstruct(graph: torch.Graph):
        # Function replaces prim::Constant containing List of Tensors with
        # prim::ListConstruct containing prim::Constant Tensors.
        assert isinstance(
            graph, torch.Graph), "Function can be called only with parameters of type torch.Graph."
        for node in graph.nodes():
            if node.kind() != "prim::Constant":
                continue
            output_type = node.output().type()
            allowed_types = [
                output_type.isSubtypeOf(torch.ListType.ofTensors()),
                output_type.isSubtypeOf(torch.ListType(
                    torch.OptionalType.ofTensor())),
            ]
            if not any(allowed_types):
                continue
            const_inputs = []
            for val in node.output().toIValue():
                const_input = graph.insertConstant(val)
                const_input.node().moveBefore(node)
                const_input.node().copyMetadata(node)
                const_inputs.append(const_input)

            replacement = graph.create("prim::ListConstruct", const_inputs)
            replacement.insertBefore(node)
            replacement.output().setType(torch.ListType.ofTensors())
            replacement.copyMetadata(node)
            node.output().replaceAllUsesWith(replacement.output())

    @staticmethod
    def _transform_optional_constants(graph: torch.Graph):
        # Function replaces prim::Constant containing torch.OptionalType with
        # prim::Constant containing torch.NoneType or type of IValue.
        assert isinstance(
            graph, torch.Graph), "Function can be called only with parameters of type torch.Graph."
        for node in graph.nodes():
            if node.kind() != "prim::Constant":
                continue
            output_type = node.output().type()
            if not isinstance(output_type, torch.OptionalType):
                continue
            value = node.output().toIValue()
            const_input = graph.insertConstant(value)
            const_input.node().moveBefore(node)
            const_input.node().copyMetadata(node)
            node.output().replaceAllUsesWith(const_input)
