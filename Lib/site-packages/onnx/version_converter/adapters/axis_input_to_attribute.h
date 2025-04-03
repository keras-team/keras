// Copyright (c) ONNX Project Contributors

/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "onnx/version_converter/adapters/adapter.h"

namespace ONNX_NAMESPACE {
namespace version_conversion {
class AxisInputToAttribute : public Adapter {
 public:
  explicit AxisInputToAttribute(
      const std::string& op_name,
      const OpSetID& initial,
      const OpSetID& target,
      size_t axis_index,
      int64_t default_axis)
      : Adapter(op_name, initial, target), axis_index(axis_index), default_axis(default_axis) {}

  Node* adapt(std::shared_ptr<Graph> graph, Node* node) const override {
    if (!HasAxisInput(node)) {
      node->i_(kaxis, this->default_axis);
      return EnsureAndReturnNode(node);
    }

    const ArrayRef<Value*>& inputs = node->inputs();
    Value* axis_val = inputs[this->axis_index];
    Node* axis_node = axis_val->node();

    if (axis_node->kind() == kConstant) {
      HandleConstantNode(node, axis_node, axis_val);
      return EnsureAndReturnNode(node);
    }

    if (graph->is_constant_initializer(axis_val)) {
      HandleInitializerNode(graph, node, axis_val);
      return EnsureAndReturnNode(node);
    }

    ONNX_ASSERTM(false, "Axis input must be a constant or initializer for promotion to attribute.");
  }

 private:
  size_t axis_index;
  int64_t default_axis;

  bool HasAxisInput(const Node* node) const {
    const ArrayRef<const Value*>& inputs = node->inputs();
    return inputs.size() > this->axis_index && inputs[this->axis_index]->node()->kind() != kUndefined;
  }

  void HandleConstantNode(Node* node, Node* axis_node, Value* axis_val) const {
    const std::vector<int64_t>& int64s = axis_node->t(kvalue).int64s();
    if (int64s.empty()) {
      std::string raw_data = axis_node->t(kvalue).raw();
      ONNX_ASSERTM(
          raw_data.size() != 0 && raw_data.size() % 8 == 0,
          "Raw Data must be non-empty and size must be a multiple of 8");
      const int64_t* raw = reinterpret_cast<const int64_t*>(raw_data.c_str());
      node->i_(kaxis, raw[0]);
    } else {
      node->i_(kaxis, int64s.at(0));
    }
    node->removeInput(this->axis_index);
    if (axis_val->uses().size() < 1) {
      axis_node->destroy();
    }
  }

  void HandleInitializerNode(std::shared_ptr<Graph> graph, Node* node, Value* axis_val) const {
    const std::string initializer_name = axis_val->uniqueName();
    for (const auto& initializer : graph->initializers()) {
      if (initializer.name() == initializer_name) {
        node->i_(kaxis, initializer.int64s().at(0));
        node->removeInput(this->axis_index);
        // Remove initializer
        if (axis_val->uses().size() < 1)
          graph->eraseInitializer(initializer_name);
        break;
      }
    }
  }

  inline Node* EnsureAndReturnNode(Node* node) const {
    ONNX_ASSERTM(node->hasAttribute(kaxis), "Axis attribute not created. This may be a bug.");
    return node;
  }
};

} // namespace version_conversion
} // namespace ONNX_NAMESPACE
