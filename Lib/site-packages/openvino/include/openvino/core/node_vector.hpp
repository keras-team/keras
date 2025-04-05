// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include "openvino/core/core_visibility.hpp"

namespace ov {

template <typename NodeType>
class Output;

class Node;

namespace op {
namespace v0 {

class Result;

}  // namespace v0
}  // namespace op

using NodeVector = std::vector<std::shared_ptr<Node>>;
using OutputVector = std::vector<Output<Node>>;
using ResultVector = std::vector<std::shared_ptr<ov::op::v0::Result>>;

OPENVINO_API
OutputVector as_output_vector(const NodeVector& args);
OPENVINO_API
NodeVector as_node_vector(const OutputVector& values);
/// Returns a ResultVector referencing values.
OPENVINO_API
ResultVector as_result_vector(const OutputVector& values);
}  // namespace ov
