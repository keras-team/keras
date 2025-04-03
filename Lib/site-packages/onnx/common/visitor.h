// Copyright (c) ONNX Project Contributors

/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include "onnx/common/common.h"
#include "onnx/onnx_pb.h"

namespace ONNX_NAMESPACE {
namespace internal {

// Visitor: A readonly visitor class for ONNX Proto objects.
// This class is restricted to Nodes, Graphs, Attributes, and Functions.
// The VisitX methods invoke ProcessX, and if that returns true, will
// continue to visit all children of the X.

struct Visitor {
  virtual void VisitGraph(const GraphProto& graph) {
    if (ProcessGraph(graph))
      for (auto& node : graph.node())
        VisitNode(node);
  }

  virtual void VisitFunction(const FunctionProto& function) {
    if (ProcessFunction(function))
      for (auto& node : function.node())
        VisitNode(node);
  }

  virtual void VisitNode(const NodeProto& node) {
    if (ProcessNode(node)) {
      for (auto& attr : node.attribute()) {
        VisitAttribute(attr);
      }
    }
  }

  virtual void VisitAttribute(const AttributeProto& attr) {
    if (ProcessAttribute(attr)) {
      if (attr.has_g()) {
        VisitGraph(attr.g());
      }
      for (auto& graph : attr.graphs())
        VisitGraph(graph);
    }
  }

  virtual bool ProcessGraph(const GraphProto& graph) {
    ONNX_UNUSED_PARAMETER(graph);
    return true;
  }

  virtual bool ProcessFunction(const FunctionProto& function) {
    ONNX_UNUSED_PARAMETER(function);
    return true;
  }

  virtual bool ProcessNode(const NodeProto& node) {
    ONNX_UNUSED_PARAMETER(node);
    return true;
  }

  virtual bool ProcessAttribute(const AttributeProto& attr) {
    ONNX_UNUSED_PARAMETER(attr);
    return true;
  }

  virtual ~Visitor() {}
};

// MutableVisitor: A version of Visitor that allows mutation of the visited objects.
struct MutableVisitor {
  virtual void VisitGraph(GraphProto* graph) {
    if (ProcessGraph(graph))
      for (auto& node : *(graph->mutable_node()))
        VisitNode(&node);
  }

  virtual void VisitFunction(FunctionProto* function) {
    if (ProcessFunction(function))
      for (auto& node : *(function->mutable_node()))
        VisitNode(&node);
  }

  virtual void VisitNode(NodeProto* node) {
    if (ProcessNode(node)) {
      for (auto& attr : *(node->mutable_attribute())) {
        VisitAttribute(&attr);
      }
    }
  }

  virtual void VisitAttribute(AttributeProto* attr) {
    if (ProcessAttribute(attr)) {
      if (attr->has_g()) {
        VisitGraph(attr->mutable_g());
      }
      for (auto& graph : *(attr->mutable_graphs()))
        VisitGraph(&graph);
    }
  }

  virtual bool ProcessGraph(GraphProto* graph) {
    ONNX_UNUSED_PARAMETER(graph);
    return true;
  }

  virtual bool ProcessFunction(FunctionProto* function) {
    ONNX_UNUSED_PARAMETER(function);
    return true;
  }

  virtual bool ProcessNode(NodeProto* node) {
    ONNX_UNUSED_PARAMETER(node);
    return true;
  }

  virtual bool ProcessAttribute(AttributeProto* attr) {
    ONNX_UNUSED_PARAMETER(attr);
    return true;
  }

  virtual ~MutableVisitor() {}
};

} // namespace internal
} // namespace ONNX_NAMESPACE
