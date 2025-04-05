// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/sink.hpp"
#include "openvino/op/util/variable_extension.hpp"

namespace ov {
namespace op {
namespace util {
class OPENVINO_API AssignBase : public Sink, public VariableExtension {
public:
    OPENVINO_OP("AssignBase", "util", ov::op::Sink);
    AssignBase() = default;
    /// \brief Constructs an AssignBase operation.
    explicit AssignBase(const OutputVector& arguments) : Sink(arguments) {}
};
}  // namespace util
}  // namespace op
}  // namespace ov
