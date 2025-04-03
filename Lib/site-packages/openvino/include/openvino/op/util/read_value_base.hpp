// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"
#include "openvino/op/util/variable_extension.hpp"

namespace ov {
namespace op {
namespace util {
class OPENVINO_API ReadValueBase : public Op, public VariableExtension {
public:
    OPENVINO_OP("ReadValueBase", "util");

    ReadValueBase() = default;

    /// \brief Constructs an AssignBase operation.
    explicit ReadValueBase(const OutputVector& arguments) : Op(arguments) {}
};
}  // namespace util
}  // namespace op
}  // namespace ov
