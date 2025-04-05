// Copyright (c) ONNX Project Contributors

/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Adapter for GroupNormalization in default domain from version 20 to 21

#pragma once

#include <memory>

#include "onnx/version_converter/adapters/adapter.h"

namespace ONNX_NAMESPACE {
namespace version_conversion {

class GroupNormalization_20_21 final : public Adapter {
 public:
  explicit GroupNormalization_20_21() : Adapter("GroupNormalization", OpSetID(20), OpSetID(21)) {}

  void transform_input(
      std::shared_ptr<Graph> graph,
      Node* node,
      int64_t input_id,
      Value* reshape0_shape,
      Value* reshape1_shape,
      Value* expand_shape) const {
    Node* reshape0 = graph->create(kReshape);
    reshape0->addInput(node->inputs()[input_id]);
    reshape0->addInput(reshape0_shape);
    reshape0->insertBefore(node);

    Node* expand = graph->create(kExpand);
    expand->addInput(reshape0->output());
    expand->addInput(expand_shape);
    expand->insertBefore(node);

    Node* reshape1 = graph->create(kReshape);
    reshape1->addInput(expand->output());
    reshape1->addInput(reshape1_shape);
    reshape1->insertBefore(node);

    node->replaceInput(input_id, reshape1->output());
  }

  void adapt_group_normalization_20_21(std::shared_ptr<Graph> graph, Node* node) const {
    // Perform following sequence of ops on scale/bias, effect is similar to numpy.repeat()
    //
    //   Shape<start=1,end=2>(input0) -- Div(Shape_out (C), num_groups)
    //                                                           |
    // Reshape(input1/2, [-1, 1]) ----------- Expand(Reshape_out, [1, Div_out]) -- Reshape(Expand_out, [-1])
    //
    // The helper function transform_input() implements the bottom row of the diagram

    // Get number of channels: C
    Symbol kShape("Shape");
    Node* C = graph->create(kShape);
    C->i_(kstart, 1);
    C->i_(kend, 2);
    C->addInput(node->inputs()[0]);
    C->insertBefore(node);

    // Get number of channels per group
    Tensor tensor_num_groups;
    tensor_num_groups.elem_type() = TensorProto_DataType_INT64;
    int64_t num_groups = node->i(knum_groups);
    tensor_num_groups.sizes() = {1};
    tensor_num_groups.int64s() = {num_groups};
    Node* constant_num_groups = graph->create(kConstant);
    constant_num_groups->t_(kvalue, tensor_num_groups);
    constant_num_groups->insertBefore(node);

    Node* div = graph->create(kDiv);
    div->addInput(C->output());
    div->addInput(constant_num_groups->output());
    div->insertBefore(node);

    // Get Expand shape: [1, Div_out]
    Tensor tensor_one;
    tensor_one.elem_type() = TensorProto_DataType_INT64;
    tensor_one.sizes() = {1};
    tensor_one.int64s() = {1};
    Node* constant_one = graph->create(kConstant);
    constant_one->t_(kvalue, tensor_one);
    constant_one->insertBefore(node);
    Node* concat = graph->create(kConcat);
    concat->i_(kaxis, 0);
    concat->addInput(constant_one->output());
    concat->addInput(div->output());
    concat->insertBefore(node);

    // Get shape of first reshape: [-1, 1]
    Tensor tensor_reshape0_shape;
    tensor_reshape0_shape.elem_type() = TensorProto_DataType_INT64;
    tensor_reshape0_shape.sizes() = {2};
    tensor_reshape0_shape.int64s() = {-1, 1};
    Node* constant_reshape0_shape = graph->create(kConstant);
    constant_reshape0_shape->t_(kvalue, tensor_reshape0_shape);
    constant_reshape0_shape->insertBefore(node);

    // Get shape of last reshape: [-1]
    Tensor tensor_reshape1_shape;
    tensor_reshape1_shape.elem_type() = TensorProto_DataType_INT64;
    tensor_reshape1_shape.sizes() = {1};
    tensor_reshape1_shape.int64s() = {-1};
    Node* constant_reshape1_shape = graph->create(kConstant);
    constant_reshape1_shape->t_(kvalue, tensor_reshape1_shape);
    constant_reshape1_shape->insertBefore(node);

    // transform scale and bias
    transform_input(
        graph, node, 1, constant_reshape0_shape->output(), constant_reshape1_shape->output(), concat->output());
    transform_input(
        graph, node, 2, constant_reshape0_shape->output(), constant_reshape1_shape->output(), concat->output());

    // Set stash_type
    node->i_(kstash_type, node->inputs()[0]->elemType());
  }

  Node* adapt(std::shared_ptr<Graph> graph, Node* node) const override {
    adapt_group_normalization_20_21(graph, node);
    return node;
  }
};

} // namespace version_conversion
} // namespace ONNX_NAMESPACE
