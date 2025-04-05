# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging

import numpy as np
from fusion_attention import AttentionMask, FusionAttention
from fusion_base import Fusion
from fusion_simplified_layernorm import FusionSimplifiedLayerNormalization, FusionSkipSimplifiedLayerNormalization
from fusion_utils import NumpyHelper
from onnx import NodeProto, TensorProto, helper
from onnx_model import OnnxModel
from onnx_model_bert import BertOnnxModel

logger = logging.getLogger(__name__)


class FusionT5Attention(FusionAttention):
    """
    Fuse T5 Attention subgraph into one Attention node.
    """

    def __init__(
        self,
        model: OnnxModel,
        hidden_size: int,
        num_heads: int,
        attention_mask: AttentionMask,
    ):
        super().__init__(
            model,
            hidden_size,
            num_heads,
            attention_mask,
            use_multi_head_attention=False,
            search_op_types=["SkipSimplifiedLayerNormalization", "Add"],
        )
        self.static_kv = 1

    def create_attention_node(
        self,
        mask_index: str,
        q_matmul: NodeProto,
        k_matmul: NodeProto,
        v_matmul: NodeProto,
        num_heads: int,
        hidden_size: int,
        input: str,
        output: str,
        add_qk_str: str,
        scale: float | None = None,
    ) -> NodeProto | None:
        """Create an Attention node.
        Args:
            mask_index (str): mask input
            q_matmul (NodeProto): MatMul node in fully connection for Q
            k_matmul (NodeProto): MatMul node in fully connection for K
            v_matmul (NodeProto): MatMul node in fully connection for V
            num_heads (int): number of attention heads. If a model is pruned, it is the number of heads after pruning.
            hidden_size (int): hidden dimension. If a model is pruned, it is the hidden dimension after pruning.
            input (str): input name
            output (str): output name
        Returns:
            Union[NodeProto, None]: the node created or None if failed.
        """
        assert num_heads > 0

        if hidden_size > 0 and (hidden_size % num_heads) != 0:
            logger.debug(f"input hidden size {hidden_size} is not a multiple of num of heads {num_heads}")
            return None

        q_weight = self.model.get_initializer(q_matmul.input[1])
        k_weight = self.model.get_initializer(k_matmul.input[1])
        v_weight = self.model.get_initializer(v_matmul.input[1])

        if q_weight is None or k_weight is None or v_weight is None:
            matmul = q_matmul if q_weight is None else k_matmul if k_weight is None else v_matmul
            print(
                f"{matmul.input[1]} is not an initializer. "
                "Please set do_constant_folding=True in torch.onnx.export to unblock attention fusion"
            )
            return None

        qw = NumpyHelper.to_array(q_weight)
        kw = NumpyHelper.to_array(k_weight)
        vw = NumpyHelper.to_array(v_weight)

        # assert q and k have same shape as expected
        assert qw.shape == kw.shape

        qw_in_size = qw.shape[0]
        kw_in_size = kw.shape[0]
        vw_in_size = vw.shape[0]

        assert qw_in_size == kw_in_size == vw_in_size

        if hidden_size > 0 and hidden_size != qw_in_size:
            logger.warning(
                f"Input hidden size ({hidden_size}) is not same as weight matrix dimension of q,k,v ({qw_in_size}). "
                "Please provide a correct input hidden size or pass in 0"
            )

        qw_out_size = np.prod(qw.shape[1:])
        qkv_weight = np.stack((qw, kw, vw), axis=1)
        qkv_weight_dim = 3 * qw_out_size

        attention_node_name = self.model.create_node_name("Attention")

        weight = helper.make_tensor(
            name=attention_node_name + "_qkv_weight",
            data_type=TensorProto.FLOAT,
            dims=[qw_in_size, qkv_weight_dim],
            vals=qkv_weight.tobytes(),
            raw=True,
        )

        self.model.add_initializer(weight, self.this_graph_name)

        attention_inputs = [
            input,
            attention_node_name + "_qkv_weight",
            "",
        ]
        if mask_index is not None:
            attention_inputs.append(mask_index)
        else:
            attention_inputs.append("")

        if add_qk_str is not None:
            attention_inputs.append("")  # no past
            attention_inputs.append(add_qk_str)

        attention_node = helper.make_node(
            "Attention",
            inputs=attention_inputs,
            outputs=[output],
            name=attention_node_name,
        )
        attention_node.domain = "com.microsoft"
        attention_node.attribute.extend([helper.make_attribute("num_heads", num_heads)])

        if scale is not None:
            attention_node.attribute.extend([helper.make_attribute("scale", scale)])

        if self.mask_filter_value is not None:
            attention_node.attribute.extend([helper.make_attribute("mask_filter_value", float(self.mask_filter_value))])

        return attention_node

    def create_mha_node(
        self,
        query: str,
        key: str,
        value: str,
        mask_index: str,
        res_pos_bias: str,
        past_key: str,
        past_value: str,
        output: str,
        present_key: str,
        present_value: str,
        num_heads: int,
        hidden_size: int,
    ) -> NodeProto | None:
        assert num_heads > 0

        if hidden_size > 0 and (hidden_size % num_heads) != 0:
            logger.debug(f"input hidden size {hidden_size} is not a multiple of num of heads {num_heads}")
            return None

        attention_node_name = self.model.create_node_name("MultiHeadAttention")
        attention_inputs = [
            query,
            "" if key is None else key,  # key
            "" if value is None else value,  # value
            "",  # bias
        ]
        if mask_index is not None:
            attention_inputs.append(mask_index)
        else:
            attention_inputs.append("")

        if res_pos_bias is not None:
            attention_inputs.append(res_pos_bias)
        else:
            attention_inputs.append("")

        if past_key is not None:
            assert past_value is not None
            attention_inputs.append(past_key)
            attention_inputs.append(past_value)

        attention_outputs = [output]
        if present_key is not None:
            assert present_value is not None
            attention_outputs.append(present_key)
            attention_outputs.append(present_value)

        attention_node = helper.make_node(
            "MultiHeadAttention",
            inputs=attention_inputs,
            outputs=attention_outputs,
            name=attention_node_name,
        )

        attention_node.domain = "com.microsoft"
        attention_node.attribute.extend([helper.make_attribute("num_heads", num_heads)])
        attention_node.attribute.extend([helper.make_attribute("scale", 1.0)])
        if self.mask_filter_value is not None:
            attention_node.attribute.extend([helper.make_attribute("mask_filter_value", float(self.mask_filter_value))])

        self.increase_counter("MultiHeadAttention")
        return attention_node

    def fuse(self, normalize_node, input_name_to_nodes, output_name_to_node):
        self.fuse_t5_encoder(normalize_node, input_name_to_nodes, output_name_to_node)
        self.fuse_t5_decoder(normalize_node, input_name_to_nodes, output_name_to_node)

    def fuse_t5_encoder(self, normalize_node, input_name_to_nodes, output_name_to_node):
        if normalize_node.op_type != "SkipSimplifiedLayerNormalization" and normalize_node.op_type != "Add":
            return

        qkv_nodes = self.model.match_parent_path(
            normalize_node, ["MatMul", "Reshape", "Transpose", "MatMul"], [1, 0, 0, 0], output_name_to_node
        )
        if qkv_nodes is None:
            return

        _, reshape_qkv, transpose_qkv, matmul_qkv = qkv_nodes

        qkv_shape_nodes = self.model.match_parent_path(
            reshape_qkv,
            ["Concat", "Unsqueeze", "Gather", "Shape"],
            [1, 0, 0, 0],
            output_name_to_node,
        )
        if qkv_shape_nodes is None:
            return
        input_shape_node = qkv_shape_nodes[-1]

        v_nodes = self.model.match_parent_path(
            matmul_qkv,
            ["Transpose", "Reshape", "MatMul"],
            [1, 0, 0],
            output_name_to_node,
        )
        if v_nodes is None:
            return
        _, reshape_v, matmul_v = v_nodes
        # todo: check reshape_v parent nodes

        qk_nodes = self.model.match_parent_path(
            matmul_qkv,
            ["Softmax", "Add", "MatMul"],
            [0, 0, 0],
            output_name_to_node,
        )
        if qk_nodes is None:
            return
        _, add_qk, matmul_qk = qk_nodes

        mask_nodes = self.model.match_parent_path(
            add_qk,
            ["Add", "Mul", "Sub", "Cast", "Unsqueeze", "Unsqueeze"],
            [1, 1, 0, 1, 0, 0],
            output_name_to_node,
        )

        is_pattern_for_one_graph_input = mask_nodes is None
        if mask_nodes is None:
            # Pattern for SD3 and Flux.
            mask_nodes = self.model.match_parent_path(
                add_qk,
                ["Add", "Slice", "Mul", "Sub", "Unsqueeze", "Unsqueeze"],
                [1, 1, 0, 0, 1, 0],
                output_name_to_node,
            )
            if mask_nodes is None:
                return
            mul_node = mask_nodes[2]
        else:
            mul_node = mask_nodes[1]

        _, mul_val = self.model.get_constant_input(mul_node)
        if mul_val is None:
            return

        if mul_val != -10000:
            self.mask_filter_value = float(mul_val)

        # If the mask is derived from shape of input_ids, it means there is no padding mask.
        mask_nodes_2 = self.model.match_parent_path(
            mask_nodes[-1],
            ["ConstantOfShape", "Concat", "Unsqueeze", "Gather", "Shape"],
            [0, 0, 0, 0, 0],
            output_name_to_node,
        )
        mask_nodes_3 = self.model.match_parent_path(
            mask_nodes[-1],
            ["ConstantOfShape", "Concat", "Unsqueeze", "Gather", "Shape"],
            [0, 0, 1, 0, 0],
            output_name_to_node,
        )
        if (
            mask_nodes_2 is not None
            and any(input.name == mask_nodes_2[-1].input[0] for input in self.model.graph().input)
            and mask_nodes_3 is not None
            and mask_nodes_2[-1].input[0] == mask_nodes_3[-1].input[0]
            and len(mask_nodes_2[1].input) == 2
        ):
            mask_index = ""
        else:
            mask_index = self.attention_mask.process_mask(mask_nodes[-1].input[0])

        res_pos_bias = None
        rpb_nodes = self.model.match_parent_path(
            add_qk,
            ["Add", "RelativePositionBias"],
            [1, 0],
        )
        if rpb_nodes is None and is_pattern_for_one_graph_input:
            # Pattern for SD3 and Flux.
            rpb_nodes = self.model.match_parent_path(
                add_qk,
                ["Add", "Slice", "RelativePositionBias"],
                [1, 0, 0],
            )
        if rpb_nodes is None:
            return

        res_pos_bias = rpb_nodes[-1].output[0]

        k_nodes = self.model.match_parent_path(
            matmul_qk,
            ["Transpose", "Reshape", "MatMul"],
            [1, 0, 0],
        )
        if k_nodes is None:
            return
        _, reshape_k, matmul_k = k_nodes
        # todo: check reshape_k parent nodes

        q_nodes = self.model.match_parent_path(
            matmul_qk,
            ["Transpose", "Reshape", "MatMul"],
            [0, 0, 0],
        )
        if q_nodes is None:
            return

        transpose_q, reshape_q, matmul_q = q_nodes
        # todo: check reshape_q parent nodes

        if matmul_q.input[0] != input_shape_node.input[0]:
            return

        q_num_heads, q_hidden_size = self.get_num_heads_and_hidden_size(reshape_q)

        new_node = self.create_attention_node(
            mask_index,
            matmul_q,
            matmul_k,
            matmul_v,
            q_num_heads,
            q_hidden_size,
            input_shape_node.input[0],
            reshape_qkv.output[0],
            res_pos_bias,
            1.0,
        )
        if new_node is None:
            return

        self.nodes_to_add.append(new_node)
        self.node_name_to_graph_name[new_node.name] = self.this_graph_name

        self.nodes_to_remove.append(reshape_qkv)
        self.prune_graph = True

    def fuse_t5_decoder(self, normalize_node, input_name_to_nodes, output_name_to_node):
        if normalize_node.op_type != "SkipSimplifiedLayerNormalization" and normalize_node.op_type != "Add":
            return

        qkv_nodes = self.model.match_parent_path(
            normalize_node,
            ["MatMul", "Reshape", "Transpose", "MatMul"],
            [1, 0, 0, 0],
        )
        if qkv_nodes is None:
            return

        _, reshape_qkv, transpose_qkv, matmul_qkv = qkv_nodes

        qkv_shape_nodes = self.model.match_parent_path(
            reshape_qkv,
            ["Concat", "Unsqueeze", "Gather", "Shape"],
            [1, 0, 0, 0],
        )
        if qkv_shape_nodes is None:
            return
        input_shape_node = qkv_shape_nodes[-1]

        value = None
        past_value = None
        present_value = None
        v_nodes = self.model.match_parent_path(
            matmul_qkv,
            ["Concat", "Transpose", "Reshape", "MatMul"],
            [1, 1, 0, 0],
        )
        if v_nodes is None:
            v_nodes = self.model.match_parent_path(
                matmul_qkv,
                ["Transpose", "Reshape", "MatMul"],
                [1, 0, 0],
            )
            if v_nodes is not None:
                transpose_v, reshape_v, matmul_v = v_nodes
                value = reshape_v.input[0]
                present_value = transpose_v.output[0]
                if "present_value" not in present_value:
                    return
                if matmul_v.input[0] != input_shape_node.input[0]:
                    self.static_kv = 1
                else:
                    self.static_kv = 0
            else:
                past_value = matmul_qkv.input[1]
                if past_value in output_name_to_node:
                    return
                if "past_value_cross" not in past_value:
                    return
                self.static_kv = 1
        else:
            concat_v, _, reshape_v, _ = v_nodes
            past_value = concat_v.input[0]
            if past_value in output_name_to_node:
                return
            if "past_value_self" not in past_value:
                return
            present_value = concat_v.output[0]
            if "present_value_self" not in present_value:
                return
            value = reshape_v.input[0]
            self.static_kv = 0

        qk_nodes = self.model.match_parent_path(
            matmul_qkv,
            ["Softmax", "Add", "MatMul"],
            [0, 0, 0],
        )
        if qk_nodes is None:
            return
        _, add_qk, matmul_qk = qk_nodes

        mask_index = None
        res_pos_bias = None
        if self.static_kv == 1:
            mask_nodes = self.model.match_parent_path(
                add_qk,
                ["Add", "Mul", "Sub", "Cast", "Unsqueeze", "Unsqueeze"],
                [1, 1, 0, 1, 0, 0],
            )
            if mask_nodes is None:
                return
            mul_node = mask_nodes[1]
            if mask_nodes[1].op_type != "Mul":
                return

            _, mul_val = self.model.get_constant_input(mul_node)
            if mul_val != -10000:
                self.mask_filter_value = mul_val

            mask_index = self.attention_mask.process_mask(mask_nodes[-1].input[0])
        else:
            rpb_nodes = self.model.match_parent_path(
                add_qk,
                ["Add", "Slice"],
                [1, 0],
            )
            if rpb_nodes is not None:
                res_pos_bias = add_qk.input[1]
            else:
                rpb_nodes = self.model.match_parent_path(
                    add_qk,
                    ["Add", "RelativePositionBias"],
                    [1, 0],
                )
                if rpb_nodes is None:
                    return
                res_pos_bias = add_qk.input[1]

        key = None
        past_key = None
        present_key = None
        if self.static_kv == 1:
            k_nodes = self.model.match_parent_path(
                matmul_qk,
                ["Transpose", "Reshape", "MatMul"],
                [1, 0, 0],
            )
            if k_nodes is not None:
                transpose_k, reshape_k, _ = k_nodes
                key = reshape_k.input[0]
                present_key_transpose_nodes = input_name_to_nodes[reshape_k.output[0]]
                for present_key_transpose_node in present_key_transpose_nodes:
                    present_key_candidate = self.model.find_graph_output(present_key_transpose_node.output[0])
                    if present_key_candidate is not None:
                        present_key = present_key_candidate.name
                        break
                if present_key is None:
                    return
                if "present_key_cross" not in present_key:
                    return
            else:
                k_nodes = self.model.match_parent_path(
                    matmul_qk,
                    ["Transpose"],
                    [1],
                )
                if k_nodes is None:
                    return
                transpose_k = k_nodes[0]

                past_key = transpose_k.input[0]
                if past_key in output_name_to_node:
                    return
                if "past_key_cross" not in past_key:
                    return
        else:
            idx, k_nodes, _ = self.model.match_parent_paths(
                matmul_qk,
                [
                    (["Transpose", "Concat", "Reshape", "MatMul"], [1, 0, 1, 0]),
                    (["Transpose", "Concat", "Transpose", "Reshape", "MatMul"], [1, 0, 1, 0, 0]),
                ],
                output_name_to_node,
            )
            past_key_transpose_node = None
            present_key_transpose_nodes = None
            if k_nodes is not None:
                concat_k, reshape_k = k_nodes[1], k_nodes[-2]
                key = reshape_k.input[0]

                if idx == 0:
                    past_key_transpose_node = output_name_to_node[concat_k.input[0]]
                    past_key = past_key_transpose_node.input[0]
                else:
                    past_key = concat_k.input[0]
                if past_key in output_name_to_node:
                    return
                if "past_key_self" not in past_key:
                    return

                if idx == 0:
                    present_key_transpose_nodes = input_name_to_nodes[concat_k.output[0]]
                    for present_key_transpose_node in present_key_transpose_nodes:
                        present_key_candidate = self.model.find_graph_output(present_key_transpose_node.output[0])
                        if present_key_candidate is not None:
                            present_key = present_key_candidate.name
                            break
                else:
                    present_key = concat_k.output[0]
                if present_key is None:
                    return
                if "present_key_self" not in present_key:
                    return
            else:
                k_nodes = self.model.match_parent_path(
                    matmul_qk,
                    ["Transpose", "Reshape", "MatMul"],
                    [1, 0, 0],
                )
                if k_nodes is None:
                    return
                _, reshape_k, _ = k_nodes
                key = reshape_k.input[0]
                present_key_transpose_nodes = input_name_to_nodes[reshape_k.output[0]]
                for present_key_transpose_node in present_key_transpose_nodes:
                    present_key_candidate = self.model.find_graph_output(present_key_transpose_node.output[0])
                    if present_key_candidate is not None:
                        present_key = present_key_candidate.name
                        break
                if present_key is None:
                    return
                if "present_key_self" not in present_key:
                    return

        q_nodes = self.model.match_parent_path(
            matmul_qk,
            ["Transpose", "Reshape", "MatMul"],
            [0, 0, 0],
        )
        if q_nodes is None:
            return

        transpose_q, reshape_q, matmul_q = q_nodes

        if matmul_q.input[0] != input_shape_node.input[0]:
            return

        q_num_heads, q_hidden_size = self.get_num_heads_and_hidden_size(reshape_q)

        if self.static_kv == 1 and past_key is not None:
            key = past_key
            value = past_value
            past_key = None
            past_value = None

        new_node = self.create_mha_node(
            matmul_q.output[0],
            key,
            value,
            mask_index,
            res_pos_bias,
            past_key,
            past_value,
            reshape_qkv.output[0],
            present_key,
            present_value,
            q_num_heads,
            q_hidden_size,
        )
        if new_node is None:
            return

        self.nodes_to_add.append(new_node)
        self.node_name_to_graph_name[new_node.name] = self.this_graph_name

        self.nodes_to_remove.append(reshape_qkv)

        self.prune_graph = True


class FusionRelativePositionBiasBlock(Fusion):
    def __init__(self, model: OnnxModel, max_distance: int):
        super().__init__(model, "RelativePositionBias", ["Add", "Slice"])
        self.max_distance = max_distance
        self.is_bidirectional = False

    def fuse(self, node, input_name_to_nodes, output_name_to_node):
        # TODO: Optimization opportunity: only last dimension of relative_position_bias is used in decoder.
        # Cuda kernel can be optimized to only compute last dimension.
        if node.op_type != "Add" and node.op_type != "Slice":
            return

        compute_bias_nodes = self.model.match_parent_path(
            node, ["Unsqueeze", "Transpose", "Gather", "Where"], [0, 0, 0, 1], output_name_to_node
        )
        if compute_bias_nodes is None:
            compute_bias_nodes = self.model.match_parent_path(
                node, ["Unsqueeze", "Transpose", "Gather", "Add", "Where"], [0, 0, 0, 1, 1], output_name_to_node
            )
            if compute_bias_nodes is None:
                return

        gather = compute_bias_nodes[2]
        where = compute_bias_nodes[-1]
        unsqueeze = compute_bias_nodes[0]

        compute_buckets_nodes = self.model.match_parent_path(
            where,
            ["Min", "ConstantOfShape", "Shape", "Add", "Cast", "Mul", "Div", "Log", "Div"],
            [2, 1, 0, 0, 0, 0, 0, 0, 0],
            output_name_to_node,
        )
        if compute_buckets_nodes is None:
            return

        # It is possible to deduce max_distance from a Div node:
        # The value of self.model.get_constant_value(compute_buckets_nodes[-3].input[1]) is close to
        #   math.log(max_distance / (relative_attention_num_buckets // (4 if is_bidirectional else 2)))
        # See https://github.com/huggingface/transformers/blob/608e163b527eaee41e650ffb9eb4c422d2679902/src/transformers/models/t5/modeling_t5.py#L397.
        # Most t5 models use max_distance=128, so we hardcode it unitl we see a model with different value.
        # TODO: maybe add a sanity check here.

        div = compute_buckets_nodes[-1]

        range_nodes = self.model.match_parent_path(
            div,
            ["Cast", "Neg", "Min", "ConstantOfShape", "Shape", "Sub", "Unsqueeze", "Range"],
            [0, 0, 0, 1, 0, 0, 0, 0],
            output_name_to_node,
        )
        if range_nodes is None:
            range_nodes = self.model.match_parent_path(
                div, ["Cast", "Abs", "Sub", "Unsqueeze", "Range"], [0, 0, 0, 0, 0], output_name_to_node
            )
            self.is_bidirectional = True
            if range_nodes is None:
                return

        range_node = range_nodes[-1]

        self.nodes_to_remove.append(unsqueeze)
        self.prune_graph = True

        node_name = self.model.create_node_name(
            "RelativePositionBias", name_prefix="RelPosBias_" + ("encoder" if self.is_bidirectional else "decoder")
        )

        table_weight_i = self.model.get_initializer(gather.input[0])
        if table_weight_i is None:
            return
        table_weight = NumpyHelper.to_array(table_weight_i)
        table_weight_t = np.transpose(table_weight)
        bias_table = helper.make_tensor(
            name=node_name + "_bias_table_weight",
            data_type=TensorProto.FLOAT,
            dims=[np.shape(table_weight)[0], np.shape(table_weight)[1]],
            vals=table_weight_t.tobytes(),
            raw=True,
        )

        self.model.add_initializer(bias_table, self.this_graph_name)
        inputs = [bias_table.name, range_node.input[1], range_node.input[1]]
        outputs = [unsqueeze.output[0]]
        rpb_node = helper.make_node(
            "RelativePositionBias",
            inputs=inputs,
            outputs=outputs,
            name=node_name,
        )
        rpb_node.domain = "com.microsoft"
        rpb_node.attribute.extend([helper.make_attribute("max_distance", self.max_distance)])
        rpb_node.attribute.extend([helper.make_attribute("is_bidirectional", self.is_bidirectional)])

        self.nodes_to_add.append(rpb_node)
        self.node_name_to_graph_name[rpb_node.name] = self.this_graph_name


class T5OnnxModel(BertOnnxModel):
    def __init__(self, model, num_heads: int = 0, hidden_size: int = 0):
        super().__init__(model, num_heads, hidden_size)
        self.attention_mask = AttentionMask(self)

        # When the model has only one input (input_ids), there is no padding mask.
        if len(self.model.graph.input) == 1:
            from fusion_options import AttentionMaskFormat

            self.attention_mask.mask_format = AttentionMaskFormat.NoMask

        self.attention_fusion = FusionT5Attention(self, self.hidden_size, self.num_heads, self.attention_mask)
        self.layer_norm_fusion = FusionSimplifiedLayerNormalization(self)
        self.skip_layer_norm_fusion = FusionSkipSimplifiedLayerNormalization(self)
        self.rpb_fusion = FusionRelativePositionBiasBlock(self, 128)

    def fuse_attention(self):
        self.attention_fusion.apply()

    def fuse_layer_norm(self):
        self.layer_norm_fusion.apply()

    def fuse_skip_layer_norm(self, shape_infer=True):
        self.skip_layer_norm_fusion.apply()

    def adjust_rel_pos_bis_length_input(self):
        # For T5 encoder, it uses complex logic to compute the query and key length when there is only one graph input (input_ids)
        # We can directly get the length from shape (the 2nd dimension) of input_ids.
        for node in self.nodes():
            if node.op_type == "RelativePositionBias":
                nodes = self.match_parent_path(
                    node,
                    [
                        "Gather",
                        "Shape",
                        "Transpose",
                        "Reshape",
                        "Concat",
                        "Unsqueeze",
                        "Gather",
                        "Shape",
                        "SimplifiedLayerNormalization",
                        "Gather",
                    ],
                    [1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                )
                # TODO: more validation on node attributes
                if nodes is not None:
                    graph_input_names = [input.name for input in self.model.graph.input]
                    if nodes[-1].input[1] in graph_input_names:
                        node_name = self.create_node_name("Shape", name_prefix="Added_Shape_")
                        shape_node = helper.make_node(
                            "Shape",
                            inputs=[nodes[-1].input[1]],
                            outputs=[node_name + "_Output"],
                            name=node_name,
                        )

                        indices_1 = helper.make_tensor(
                            name="Constant_Index_1",
                            data_type=TensorProto.INT64,
                            dims=[1],  # Shape of the tensor
                            vals=[1],  # Tensor values
                        )
                        self.add_initializer(indices_1)

                        gather = helper.make_node(
                            "Gather",
                            inputs=[node_name + "_Output", "Constant_Index_1"],
                            outputs=[node_name + "_Output_Gather_1"],
                            name=self.create_node_name("Gather", name_prefix="Added_Gather_"),
                            axis=0,
                        )

                        self.add_node(shape_node)
                        self.add_node(gather)
                        node.input[1] = node_name + "_Output_Gather_1"
                        node.input[2] = node_name + "_Output_Gather_1"

                break

    # Remove get_extended_attention_mask() since it generates all zeros.
    def remove_extended_mask_decoder_init(self):
        nodes_to_remove = []
        for node in self.nodes():
            if node.op_type == "Add":
                extended_mask_nodes = self.match_parent_path(
                    node,
                    [
                        "Mul",
                        "Sub",
                        "Mul",
                        "Unsqueeze",
                        "Cast",
                        "LessOrEqual",
                        "Tile",
                        "Concat",
                        "Unsqueeze",
                        "Gather",
                        "Shape",
                    ],
                    [1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
                )
                if extended_mask_nodes is None:
                    continue

                rpb_nodes = self.match_parent_path(node, ["RelativePositionBias"], [0])
                if rpb_nodes is None:
                    continue

                rpb_node = rpb_nodes[0]
                rpb_node.output[0] = node.output[0]

                nodes_to_remove.extend(extended_mask_nodes)
                nodes_to_remove.append(node)
                self.remove_nodes(nodes_to_remove)

    def remove_extended_mask_decoder(self):
        nodes_to_remove = []
        for node in self.nodes():
            if node.op_type == "Add":
                extended_mask_nodes = self.match_parent_path(
                    node,
                    [
                        "Mul",
                        "Sub",
                        "Mul",
                        "Unsqueeze",
                        "Concat",
                        "Cast",
                        "LessOrEqual",
                        "Tile",
                        "Concat",
                        "Unsqueeze",
                        "Gather",
                        "Shape",
                    ],
                    [1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0],
                )
                if extended_mask_nodes is None:
                    continue

                rpb_nodes = self.match_parent_path(node, ["Slice", "RelativePositionBias"], [0, 0])
                if rpb_nodes is None:
                    continue

                rpb_node = rpb_nodes[0]
                rpb_node.output[0] = node.output[0]

                nodes_to_remove.extend(extended_mask_nodes)
                nodes_to_remove.append(node)
                self.remove_nodes(nodes_to_remove)

    def preprocess(self):
        self.adjust_reshape_and_expand()
        self.rpb_fusion.apply()

    def postprocess(self):
        # remove get_extended_attention_mask() since it generates all zeros.
        self.remove_extended_mask_decoder_init()
        self.remove_extended_mask_decoder()
        self.adjust_rel_pos_bis_length_input()

        self.prune_graph()
