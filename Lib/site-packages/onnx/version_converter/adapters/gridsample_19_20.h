// Copyright (c) ONNX Project Contributors

/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Adapter for GridSample in default domain from version 19 to 20

#pragma once

#include <memory>

namespace ONNX_NAMESPACE {
namespace version_conversion {

class GridSample_19_20 final : public Adapter {
 public:
  explicit GridSample_19_20() : Adapter("GridSample", OpSetID(19), OpSetID(20)) {}

  void adapt_gridsample_19_20(std::shared_ptr<Graph>, Node* node) const {
    if (node->hasAttribute(kmode) && (node->s(kmode) == "bilinear")) {
      node->s_(kmode, "linear");
    }
    if (node->hasAttribute(kmode) && (node->s(kmode) == "bicubic")) {
      node->s_(kmode, "cubic");
    }
  }

  Node* adapt(std::shared_ptr<Graph> graph, Node* node) const override {
    adapt_gridsample_19_20(graph, node);
    return node;
  }
};

} // namespace version_conversion
} // namespace ONNX_NAMESPACE
