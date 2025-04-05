// Copyright (c) ONNX Project Contributors

/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include <iostream>

#include "gtest/gtest.h"
#include "onnx/common/ir.h"
#include "onnx/common/ir_pb_converter.h"
#include "onnx/defs/printer.h"

namespace ONNX_NAMESPACE {
namespace Test {

static bool IsValidIdentifier(const std::string& name) {
  if (name.empty()) {
    return false;
  }
  if (!isalpha(name[0]) && name[0] != '_') {
    return false;
  }
  for (size_t i = 1; i < name.size(); ++i) {
    if (!isalnum(name[i]) && name[i] != '_') {
      return false;
    }
  }
  return true;
}

TEST(IR, ValidIdentifierTest) {
  Graph* g = new Graph();
  g->setName("test");
  Value* x = g->addInput();
  x->setUniqueName("x");
  x->setElemType(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  x->setSizes({Dimension("M"), Dimension("N")});
  Node* node1 = g->create(kNeg, 1);
  node1->addInput(x);
  g->appendNode(node1);
  Value* temp1 = node1->outputs()[0];
  Node* node2 = g->create(kNeg, 1);
  node2->addInput(temp1);
  g->appendNode(node2);
  Value* y = node2->outputs()[0];
  g->registerOutput(y);

  ModelProto model;
  ExportModelProto(&model, std::shared_ptr<Graph>(g));

  for (auto& node : model.graph().node()) {
    for (auto& name : node.output()) {
      EXPECT_TRUE(IsValidIdentifier(name));
    }
  }
}

} // namespace Test
} // namespace ONNX_NAMESPACE
