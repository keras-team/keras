// Copyright (c) ONNX Project Contributors

/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include "onnx/common/visitor.h"

namespace ONNX_NAMESPACE {
namespace internal { // internal/private API

using AttributeMap = std::unordered_map<std::string, const AttributeProto*>;

// Class for binding formal attribute-parameters (in a node or graph) to their values.

class AttributeBinder : public MutableVisitor {
 public:
  AttributeBinder(const AttributeMap& attr_map) : attr_map_(attr_map) {}

  // Binding a formal attribute-parameter to a value may, as a special case, also
  // remove the attribute from the list of attributes of a node (when the attribute
  // has no specified value). Hence, we need to do the processing at a Node level
  // rather than an attribute level.
  void VisitNode(NodeProto* node) override {
    auto& attributes = *node->mutable_attribute();
    for (auto attr_iter = attributes.begin(); attr_iter != attributes.end();) {
      auto& attr = *attr_iter;
      if (!attr.ref_attr_name().empty()) {
        // Attribute-references must be replaced by the corresponding attribute-value in the call-node
        // if the call-node contains the attribute. Otherwise, this attribute must be removed.
        auto it = attr_map_.find(attr.ref_attr_name());
        if (it != attr_map_.end()) {
          const AttributeProto* replacement = it->second;
          // Copy value of attribute, but retain original name:
          std::string name = attr.name();
          attr = *replacement;
          attr.set_name(name);
          ++attr_iter;
        } else {
          attr_iter = attributes.erase(attr_iter);
        }
      } else {
        // For regular attributes, we process subgraphs, if present, recursively.
        VisitAttribute(&attr);
        ++attr_iter;
      }
    }
  }

  // Updates a FunctionProto by replacing all attribute-references with the corresponding
  // attribute-values in the call-node, if present. Otherwise, the attribute is removed.
  static void BindAttributes(const NodeProto& callnode, FunctionProto& callee) {
    AttributeMap map;
    for (auto& attr : callnode.attribute()) {
      map[attr.name()] = &attr;
    }
    AttributeBinder attr_binder(map);
    attr_binder.VisitFunction(&callee);
  }

 private:
  const AttributeMap& attr_map_;
};

} // namespace internal
} // namespace ONNX_NAMESPACE
