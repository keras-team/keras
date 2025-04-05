// Copyright (c) ONNX Project Contributors

/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "onnx/version_converter/adapters/adapter.h"

namespace ONNX_NAMESPACE {
namespace version_conversion {

class AxisAttributeToInput : public Adapter {
 public:
  AxisAttributeToInput(
      const std::string& op_name,
      const OpSetID& initial,
      const OpSetID& target,
      size_t axis_index,
      int64_t default_axis)
      : Adapter(op_name, initial, target), axis_index(axis_index), default_axis(default_axis) {}

  Node* adapt(std::shared_ptr<Graph> graph, Node* node) const override {
    if (node->hasAttribute(kaxis)) {
      AttrToInput(graph, node, node->i(kaxis), this->axis_index);
      node->removeAttribute(kaxis);
      return node;
    }

    // Fill in the default value for axis
    AttrToInput(graph, node, default_axis, this->axis_index);
    return node;
  }

 private:
  size_t axis_index;
  int64_t default_axis;

  void AttrToInput(std::shared_ptr<Graph> graph, Node* node, int64_t axis, size_t axis_index) const {
    const ArrayRef<Value*>& inputs = node->inputs();

    // Add the optional inputs if they don't exist
    for (size_t i = inputs.size(); i < axis_index; ++i) {
      Node* empty_input = graph->create(kUndefined);
      empty_input->insertBefore(node);
      node->addInput(empty_input->output());
    }

    // Add the axis input
    Node* constant = CreateAxisInput(graph, node, axis);
    node->addInput(constant->output());
  }

  Node* CreateAxisInput(std::shared_ptr<Graph> graph, Node* node, int64_t axis) const {
    Tensor t;
    t.elem_type() = TensorProto_DataType_INT64;
    t.sizes() = std::vector<int64_t>{};
    auto& data = t.int64s();
    data.emplace_back(axis);

    Node* constant = graph->create(kConstant);
    constant->insertBefore(node);
    constant->t_(kvalue, t);
    return constant;
  }
};

} // namespace version_conversion
} // namespace ONNX_NAMESPACE
