# SPDX-License-Identifier: Apache-2.0

""" tf2onnx mapping functions for string ops using contrib ops domain. """
import io
import json
import logging
import numpy as np
from onnx.numpy_helper import to_array
from onnx.onnx_pb import TensorProto
from onnx.helper import make_attribute
from tf2onnx import constants, handler
from tf2onnx.handler import tf_op
from tf2onnx import utils
from tf2onnx.graph_builder import GraphBuilder

logger = logging.getLogger(__name__)

# pylint: disable=unused-argument,missing-docstring

@tf_op(["StringSplit", "StringSplitV2"], domain=constants.CONTRIB_OPS_DOMAIN)
class StringOps:
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        if node.type == "StringSplit":
            skip_empty = node.get_attr_value('skip_empty', True)
        else:
            skip_empty = False
        node.type = "StringSplit"
        node.domain = constants.CONTRIB_OPS_DOMAIN
        for a in list(node.attr.keys()):
            del node.attr[a]
        unsqueeze_node = GraphBuilder(ctx).make_unsqueeze({'data': node.input[1], 'axes': [0]}, return_node=True)

        skip_empty_const = ctx.make_const(utils.make_name('skip_empty_const'), np.array([skip_empty], bool))
        ctx.replace_inputs(node, [node.input[0], unsqueeze_node.output[0], skip_empty_const.output[0]])

@tf_op("StringToHashBucketFast", domain=constants.CONTRIB_OPS_DOMAIN)
class StringToHashBucketFast:
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        node.domain = constants.CONTRIB_OPS_DOMAIN
        num_buckets = node.get_attr_int('num_buckets')
        num_buckets_const = ctx.make_const(utils.make_name('num_buckets'), np.array([num_buckets], dtype=np.int64))
        ctx.replace_inputs(node, [node.input[0], num_buckets_const.output[0]])
        del node.attr['num_buckets']

@tf_op("StaticRegexReplace", domain=constants.CONTRIB_OPS_DOMAIN)
class StaticRegexReplace:
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        node.domain = constants.CONTRIB_OPS_DOMAIN
        node.type = "StringRegexReplace"
        pattern = node.get_attr_str("pattern")
        rewrite = node.get_attr_str("rewrite")
        utils.make_sure(node.get_attr_value("replace_global") != 0,
                        "Can not convert StaticRegexReplace if replace_global is False")
        pattern_node = ctx.make_const(utils.make_name("pattern"), np.array([pattern], object))
        rewrite_node = ctx.make_const(utils.make_name("rewrite"), np.array([rewrite], object))
        del node.attr["pattern"]
        del node.attr["rewrite"]
        del node.attr["replace_global"]
        ctx.replace_inputs(node, [node.input[0], pattern_node.output[0], rewrite_node.output[0]])

@tf_op("StringJoin", domain=constants.CONTRIB_OPS_DOMAIN)
class StringJoin:
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        node.domain = constants.CONTRIB_OPS_DOMAIN
        separator = node.get_attr_value("separator")
        if separator is None:
            separator = b''
        separator = separator.decode('UTF-8')
        separator_node = ctx.make_const(utils.make_name("separator"), np.array([separator], object))
        axis_node = ctx.make_const(utils.make_name("axis"), np.array([0], np.int64))
        inps_with_shapes = [i for i in node.input if ctx.get_shape(i) != []]
        shape_node = None
        if 0 < len(inps_with_shapes) < len(node.input):
            shape_node = ctx.make_node("Shape", [inps_with_shapes[0]])
        unsqueezes = []
        for inp in node.input:
            if ctx.get_shape(inp) == [] and shape_node is not None:
                utils.make_sure(ctx.opset >= 8, "Opset 8 required for Expand node for StringJoin")
                expand_node = ctx.make_node("Expand", [inp, shape_node.output[0]])
                inp = expand_node.output[0]
            unsqueeze_node = GraphBuilder(ctx).make_unsqueeze({'data': inp, 'axes': [0]})
            unsqueezes.append(unsqueeze_node)
        stack_node = ctx.make_node("Concat", unsqueezes, attr={'axis': 0})
        ctx.replace_inputs(node, [stack_node.output[0], separator_node.output[0], axis_node.output[0]])

@tf_op("ReduceJoin", domain=constants.CONTRIB_OPS_DOMAIN)
class ReduceJoin:
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        node.domain = constants.CONTRIB_OPS_DOMAIN
        node.type = "StringJoin"
        axis_node = ctx.get_node_by_output(node.input[1])
        axis = axis_node.get_attr_value('value')
        utils.make_sure(axis.dims in [[], [1]], "Only a single axis is supported for ReduceJoin node")
        axis = to_array(axis)
        new_axis_node = ctx.make_const(utils.make_name("axis"), np.array(axis, np.int64).reshape((1)))
        separator = node.get_attr_value("separator")
        if isinstance(separator, bytes):
            separator = separator.decode()
        separator_node = ctx.make_const(utils.make_name("separator"), np.array([separator], object))
        ctx.replace_inputs(node, [node.input[0], separator_node.output[0], new_axis_node.output[0]])
        keep_dims = node.get_attr_value("keep_dims")
        if keep_dims:
            unsqueeze_node = GraphBuilder(ctx).make_unsqueeze(
                {'data': node.output[0], 'axes': [-1]},
                name=node.name + '/Unsqueeze'
                )
            ctx.insert_node_on_output(ctx.get_node_by_output(unsqueeze_node))

@tf_op(["Equal", "NotEqual"], domain=constants.CONTRIB_OPS_DOMAIN)
class StringEqual:
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        dtype = ctx.get_dtype(node.input[0])
        if dtype != TensorProto.STRING:
            # Fallback to normal domain conversion
            func, _ = handler.tf_op.find_effective_op(node.type, constants.ONNX_DOMAIN)
            func(ctx, node, **kwargs)
            return

        need_not = node.type == "NotEqual"
        node.type = "StringEqual"
        node.domain = constants.CONTRIB_OPS_DOMAIN
        if need_not:
            output_name = node.output[0]
            not_node = ctx.insert_new_node_on_output("Not", output_name, name=utils.make_name(node.name))
            ctx.copy_shape(output_name, not_node.output[0])
            ctx.copy_dtype(output_name, not_node.output[0])

@tf_op(["StringLower", "StringUpper"])
class StringLower:
    @classmethod
    def version_10(cls, ctx, node, **kwargs):
        if node.type == "StringLower":
            case_action = "LOWER"
        else:
            case_action = "UPPER"
        node.type = "StringNormalizer"
        str_input = node.input[0]
        rank = ctx.get_rank(node.input[0])
        shape = ctx.get_shape(node.input[0])
        if rank != 1:
            ctx.insert_new_node_on_input(node, "Flatten", node.input[0], axis=0)
            ctx.update_node_shape_dtype(node, override=True)
        node.set_attr("case_change_action", case_action)
        if rank != 1:
            if shape is None or -1 in shape:
                new_shape = ctx.make_node("Shape", [str_input]).output[0]
            else:
                new_shape = ctx.make_const(utils.make_name("shape"), np.array(shape, np.int64)).output[0]
            ctx.insert_new_node_on_output("Reshape", node.output[0], inputs=[node.output[0], new_shape])

@tf_op("SentencepieceOp", domain=constants.CONTRIB_OPS_DOMAIN)
class SentencepieceOp:
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        # This op will be removed when its consumer is converted
        pass

@tf_op("SentencepieceTokenizeOp", domain=constants.CONTRIB_OPS_DOMAIN)
class SentencepieceTokenizeOp:
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        node.domain = constants.CONTRIB_OPS_DOMAIN
        input_node = node.inputs[0]
        utils.make_sure(input_node.type == "SentencepieceOp", "Input 0 to node %s is not SentencepieceOp", node.name)
        ctx.remove_input(node, node.input[0], 0)

        nbest_size_cast = ctx.make_node("Cast", [node.input[1]], attr={'to': TensorProto.INT64}).output[0]
        ctx.replace_input(node, node.input[1], nbest_size_cast, 1)
        for i in range(1, len(node.input)):
            unsqueeze = GraphBuilder(ctx).make_unsqueeze({'data': node.input[i], 'axes': [0]})
            ctx.replace_input(node, node.input[i], unsqueeze, i)
        node.set_attr("model", input_node.attr['model'].s)
        node.type = "SentencepieceTokenizer"
        if ctx.is_safe_to_remove_nodes([input_node]):
            ctx.remove_node(input_node.name)

@tf_op("RegexSplitWithOffsets", domain=constants.CONTRIB_OPS_DOMAIN)
class RegexSplitWithOffsetsOp:
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        node.domain = constants.CONTRIB_OPS_DOMAIN
        node.type = "StringRegexSplitWithOffsets"

@tf_op("WordpieceTokenizeWithOffsets", domain=constants.CONTRIB_OPS_DOMAIN)
class WordpieceTokenizeWithOffsetsOp:
    @classmethod
    def version_1(cls, ctx, node, initialized_tables=None, **kwargs):
        node.domain = constants.CONTRIB_OPS_DOMAIN
        node.type = "WordpieceTokenizer"
        utils.make_sure(len(node.input) == 2,
                        "[WordpieceTokenizeWithOffsetsOp] Expecting 2 inputs not %r.", len(node.input))
        utils.make_sure(initialized_tables is not None,
                        "[WordpieceTokenizeWithOffsetsOp] initialized_tables cannot be None.", len(node.input))
        parent = ctx.get_node_by_output(node.input[1])
        while parent.type == 'Identity':
            parent = ctx.get_node_by_output(parent.input[0])
        utils.make_sure(parent is not None,
                        "[WordpieceTokenizeWithOffsetsOp] Unable to extract the vocabulary")
        ressource = parent.get_attr_value('shared_name')
        table = initialized_tables[ressource]
        if isinstance(table, tuple):
            table = table[0]
        mapping = {}
        for i, word in enumerate(table):
            if isinstance(word, bytes):
                word = word.decode('utf-8')
            mapping[word] = i
        st = io.StringIO()
        json.dump(mapping, st, separators=(',', ':'))
        node.attr['vocab'] = make_attribute('vocab', st.getvalue())

        positions = ctx.make_const(utils.make_name("empty"), np.array([], np.int64))
        ctx.replace_inputs(node, [node.input[0], positions.output[0]])
