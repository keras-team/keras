// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/node.hpp"

namespace ov {
namespace op {
namespace util {
std::tuple<element::Type, PartialShape> validate_and_infer_elementwise_args(Node* node);
}
}  // namespace op
}  // namespace ov
