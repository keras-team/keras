// Copyright (c) ONNX Project Contributors

/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include <iostream>

#include "gtest/gtest.h"
#include "onnx/checker.h"
#include "onnx/common/constants.h"
#include "onnx/defs/parser.h"
#include "onnx/defs/printer.h"
#include "onnx/defs/schema.h"
#include "onnx/inliner/inliner.h"
#include "onnx/shape_inference/implementation.h"

namespace ONNX_NAMESPACE {
namespace Test {

static void InlineFunctions(ModelProto& model, const char* input, const inliner::FunctionIdSet* to_inline = nullptr) {
  OnnxParser parser(input);
  auto status = parser.Parse(model);
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();
  EXPECT_TRUE(parser.EndOfInput()) << "Extra unparsed input unexpected.";

  checker::check_model(model, false, true);
  shape_inference::InferShapes(model);

  // std::cout << "Before inlining:\n" << ProtoToString(model) << "\n";
  if (to_inline != nullptr)
    inliner::InlineSelectedFunctions(model, *to_inline);
  else
    inliner::InlineLocalFunctions(model, true);
  // std::cout << "After inlining:\n" << ProtoToString(model) << "\n";

  // The following will ensure basic safety checks hold after inlining, including
  // absence of duplicate names (multiple assignments to same name).
  checker::check_model(model, true, true);
}

TEST(FunctionInliner, BasicTest) {
  const char* code = R"ONNX(
<
  ir_version: 8,
  opset_import: [ "" : 10, "local" : 1 ]
>
agraph (float[N, 128] X, float[128,10] W, float[10] B) => (float[N, 10] C)
{
  T = local.foo (X, W, B)
  C = local.square(T)
}

<
  opset_import: [ "" : 10 ],
  domain: "local",
  doc_string: "Function foo."
>
foo (x, w, b) => (c) {
  T = MatMul(x, w)
  S = Add(T, b)
  c = Softmax(S)
}

<
  opset_import: [ "" : 10 ],
  domain: "local",
  doc_string: "Function square."
>
square (x) => (y) {
  y = Mul (x, x)
}
)ONNX";

  ModelProto model;
  InlineFunctions(model, code);
  auto num_nodes = model.graph().node_size();
  ASSERT_EQ(num_nodes, 4);
  auto num_functions = model.functions_size();
  ASSERT_EQ(num_functions, 0);
}

// Test that inlining processes subgraphs.
TEST(FunctionInliner, SubgraphTest) {
  const char* code = R"ONNX(
<
  ir_version: 8,
  opset_import: [ "" : 10, "local" : 1 ]
>
agraph (bool cond, float[N] X) => (float[N] Y)
{
  Y = If (cond) <
    then_branch = then_graph () => (y) {
        y = local.square (X)
    },
    else_branch = else_graph () => (y) {
        y = local.square (X)
    }
  >
}

<
  opset_import: [ "" : 10 ],
  domain: "local",
  doc_string: "Function square."
>
square (x) => (y) {
  y = Mul (x, x)
}
)ONNX";

  ModelProto model;
  InlineFunctions(model, code);
  auto& if_node = model.graph().node(0);
  auto& graph1 = if_node.attribute(0).g();
  ASSERT_EQ(graph1.node(0).op_type(), "Mul");
  auto& graph2 = if_node.attribute(1).g();
  ASSERT_EQ(graph2.node(0).op_type(), "Mul");
  auto num_functions = model.functions_size();
  ASSERT_EQ(num_functions, 0);
}

TEST(FunctionInliner, Nested) {
  const char* code = R"ONNX(
<ir_version: 8, opset_import: [ "" : 17, "local" : 1 ]>
agraph (float[N] X) => (float[N] Y)
{
  Y = local.foo (X)
}

<opset_import: [ "" : 17, "local" : 1 ], domain: "local">
foo (x) => (y) {
  temp = Add(x, x)
  y = local.bar(temp)
}

<opset_import: [ "" : 17 ], domain: "local">
bar (x) => (y) {
  y = Mul (x, x)
}
)ONNX";

  ModelProto model;
  InlineFunctions(model, code);
  auto num_nodes = model.graph().node_size();
  ASSERT_EQ(num_nodes, 2);
  auto num_functions = model.functions_size();
  ASSERT_EQ(num_functions, 0);
}

TEST(FunctionInliner, Renaming) {
  const char* code = R"ONNX(
<ir_version: 8, opset_import: [ "" : 17, "local" : 1 ]>
agraph (float[N] X) => (float[N] Y)
{
  temp = local.foo (X)
  temp__1 = Mul (temp, temp)
  Y = Abs (temp__1)
}

<opset_import: [ "" : 17, "local" : 1 ], domain: "local">
foo (x) => (y) {
  temp = Add(x, x)
  y = Neg (temp)
}
)ONNX";

  ModelProto model;
  // Check that renaming handles accidental collision of names: when "temp" in "foo" is
  // inlined, it will be renamed into something distinct from "temp" and "temp__1" as
  // both these names occur in the main graph.
  InlineFunctions(model, code);
}

TEST(FunctionInliner, ValueInfoPropagation) {
  const char* code = R"ONNX(
<ir_version: 10, opset_import: [ "" : 17, "local" : 1 ]>
agraph (float[N] X) => (float[N] Y)
{
  result = local.foo (X)
  Y = Abs (result)
}

<opset_import: [ "" : 17, "local" : 1 ], domain: "local">
foo (x) => (y)
<float[N] temp> {
  temp = Add(x, x)
  y = Neg (temp)
}
)ONNX";

  ModelProto model;
  InlineFunctions(model, code);
  // Check that valueinfo is propagated fron function to main graph.
  auto& graph = model.graph();
  auto& temp_new_name = graph.node(0).output(0);
  auto& valueinfos = graph.value_info();
  for (auto& valueinfo : valueinfos) {
    if (valueinfo.name() == temp_new_name) {
      ASSERT_TRUE(valueinfo.has_type());
      ASSERT_TRUE(valueinfo.type().has_tensor_type());
      ASSERT_TRUE(valueinfo.type().tensor_type().has_shape());
      ASSERT_TRUE(valueinfo.type().tensor_type().shape().dim_size() == 1);
      return;
    }
  }
  ASSERT_TRUE(false) << "ValueInfo not found";
}

TEST(FunctionInliner, TwoCallsToSameFunction) {
  const char* code = R"ONNX(
<ir_version: 8, opset_import: [ "" : 17, "local" : 1 ]>
agraph (float[N] X) => (float[N] Y)
{
  temp = local.foo (X)
  Y = local.foo (temp)
}

<opset_import: [ "" : 17, "local" : 1 ], domain: "local">
foo (x) => (y) {
  temp = Add(x, x)
  y = Neg (temp)
}
)ONNX";

  ModelProto model;
  // The call below will check that multiple assignments to same name does not happen
  // after inlining two calls to same function.
  InlineFunctions(model, code);
}

TEST(FunctionInliner, OpsetMismatch) {
  const char* code = R"ONNX(
<ir_version: 8, opset_import: [ "" : 17, "local" : 1 ]>
agraph (float[N] X) => (float[N] Y)
{
  temp = local.foo (X)
  Y = local.bar (temp)
}

<opset_import: [ "" : 18], domain: "local">
foo (x) => (y) {
  y = Add(x, x)
}

<opset_import: [ "" : 17], domain: "local">
bar (x) => (y) {
  y = Add(x, x)
}
)ONNX";

  ModelProto model;
  InlineFunctions(model, code);

  // The first node's call, to foo, must be inlined.
  auto& first_node = model.graph().node(0);
  // Check that it is a call to Add
  ASSERT_EQ(first_node.op_type(), "Add");

  // The second node's call, to bar, must be inlined.
  auto& second_node = model.graph().node(1);
  // Check that it is a call to Add
  ASSERT_EQ(second_node.op_type(), "Add");

  ASSERT_EQ(model.functions_size(), 0);
}

TEST(FunctionInliner, SelectiveInlining) {
  const char* code = R"ONNX(
<ir_version: 8, opset_import: [ "" : 17, "local" : 1 ]>
agraph (float[N] X) => (float[N] Y)
{
  temp = local.foo (X)
  Y = local.bar (temp)
}

<opset_import: [ "" : 17], domain: "local">
foo (x) => (y) {
  y = Add(x, x)
}

<opset_import: [ "" : 17, "local" : 1], domain: "local">
bar (x) => (y) {
  y = local.foo(x)
}
)ONNX";

  ModelProto model;
  inliner::FunctionIdVector to_inline = {{"local", "foo"}};
  auto to_inline_set = inliner::FunctionIdSet::Create(std::move(to_inline));
  InlineFunctions(model, code, to_inline_set.get());

  // The first node's call, to foo, must be inlined.
  auto& first_node = model.graph().node(0);
  // Check that it is a call to Add
  ASSERT_EQ(first_node.op_type(), "Add");

  // The second node's call, to bar, must not be inlined.
  auto& second_node = model.graph().node(1);
  // Check that it is a call to bar
  ASSERT_EQ(second_node.op_type(), "bar");

  // foo will be removed, bar will remain, in model.functions()
  ASSERT_EQ(model.functions_size(), 1);

  auto& bar_node = model.functions(0).node(0);
  // Check that it is a call to Add, due to inlining
  // the call to foo in bar.
  ASSERT_EQ(bar_node.op_type(), "Add");
}

TEST(FunctionInliner, VersionConversion) {
  const char* code = R"ONNX(
<ir_version: 8, opset_import: [ "" : 18, "local" : 1 ]>
agraph (float[N,M] X) => (float[N,M] Y)
{
  Y = local.foo (X)
}

<opset_import: [ "" : 17], domain: "local">
foo (x) => (y) {
  y = ReduceLogSum <axes = [0]> (x)
}
)ONNX";

  ModelProto model;
  InlineFunctions(model, code);
  // Inlining ReduceLogSum (version 17) should convert it to ReduceLogSum (version 18)
  // by promoting axes from attribute to input.
  auto& node = model.graph().node(1);
  ASSERT_EQ(node.op_type(), "ReduceLogSum");
  ASSERT_EQ(node.input_size(), 2);
  ASSERT_EQ(node.attribute_size(), 0);
}

TEST(FunctionInliner, NestedVersionConversion) {
  const char* code = R"ONNX(
<ir_version: 8, opset_import: [ "" : 18, "local" : 1 ]>
agraph (float[N,M] X) => (float[N,M] Y)
{
  Y = local.foo (X)
}

<opset_import: [ "" : 17, "local" : 1], domain: "local">
foo (x) => (y) {
  t = ReduceLogSum <axes = [0]> (x)
  y = local.bar (t)
}

<opset_import: [ "" : 17], domain: "local">
bar (x) => (y) {
  y = ReduceLogSum <axes = [1]> (x)
}
)ONNX";

  ModelProto model;
  InlineFunctions(model, code);
  // Inlining ReduceLogSum (version 17) should convert it to ReduceLogSum (version 18)
  // by promoting axes from attribute to input, with a preceding Constant node for
  // the axes value.
  // Check that both ReduceLogSum nodes have been converted.
  ASSERT_EQ(model.graph().node_size(), 4);
  ASSERT_EQ(model.graph().node(0).op_type(), "Constant");
  auto& node = model.graph().node(1);
  ASSERT_EQ(node.op_type(), "ReduceLogSum");
  ASSERT_EQ(node.input_size(), 2);
  ASSERT_EQ(node.attribute_size(), 0);
  ASSERT_EQ(model.graph().node(2).op_type(), "Constant");
  auto node2 = model.graph().node(3);
  ASSERT_EQ(node2.op_type(), "ReduceLogSum");
  ASSERT_EQ(node2.input_size(), 2);
  ASSERT_EQ(node2.attribute_size(), 0);
}

} // namespace Test
} // namespace ONNX_NAMESPACE
