// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace util {
class OPENVINO_API ShapeOfBase : public Op {
public:
    OPENVINO_OP("ShapeOfBase", "util");

    ShapeOfBase() = default;

    /// \brief Constructs an ShapeOfBase operation.
    explicit ShapeOfBase(const OutputVector& arguments) : Op(arguments) {}
};
}  // namespace util
}  // namespace op
}  // namespace ov
