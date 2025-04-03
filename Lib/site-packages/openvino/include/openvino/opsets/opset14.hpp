// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/ops.hpp"

namespace ov {
namespace opset14 {
#define _OPENVINO_OP_REG(a, b) using b::a;
#include "openvino/opsets/opset14_tbl.hpp"
#undef _OPENVINO_OP_REG
}  // namespace opset14
}  // namespace ov
