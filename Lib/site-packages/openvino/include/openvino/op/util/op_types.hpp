// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "openvino/core/core_visibility.hpp"
#include "openvino/core/node.hpp"

namespace ov {
namespace op {
namespace util {
OPENVINO_API
bool is_unary_elementwise_arithmetic(const Node* node);
OPENVINO_API
bool is_binary_elementwise_arithmetic(const Node* node);
OPENVINO_API
bool is_binary_elementwise_comparison(const Node* node);
OPENVINO_API
bool is_binary_elementwise_logical(const Node* node);

OPENVINO_API
bool supports_auto_broadcast(const Node* node);

OPENVINO_API
bool is_op(const Node* node);
OPENVINO_API
bool is_parameter(const Node* node);
OPENVINO_API
bool is_output(const Node* node);
OPENVINO_API
bool is_sink(const Node* node);
OPENVINO_API
bool is_constant(const Node* node);
OPENVINO_API
bool is_commutative(const Node* node);

OPENVINO_API
bool is_unary_elementwise_arithmetic(const std::shared_ptr<Node>& node);
OPENVINO_API
bool is_binary_elementwise_arithmetic(const std::shared_ptr<Node>& node);
OPENVINO_API
bool is_binary_elementwise_comparison(const std::shared_ptr<Node>& node);
OPENVINO_API
bool is_binary_elementwise_logical(const std::shared_ptr<Node>& node);

OPENVINO_API
bool supports_auto_broadcast(const std::shared_ptr<Node>& node);

OPENVINO_API
bool is_op(const std::shared_ptr<Node>& node);
OPENVINO_API
bool is_parameter(const std::shared_ptr<Node>& node);
OPENVINO_API
bool is_output(const std::shared_ptr<Node>& node);
OPENVINO_API
bool is_sink(const std::shared_ptr<Node>& node);
OPENVINO_API
bool is_constant(const std::shared_ptr<Node>& node);
OPENVINO_API
bool is_commutative(const std::shared_ptr<Node>& node);
}  // namespace util
}  // namespace op
}  // namespace ov
