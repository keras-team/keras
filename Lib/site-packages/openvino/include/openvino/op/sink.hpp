// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

#include "openvino/op/op.hpp"

namespace ov {
namespace op {
/// \brief Root of nodes that can be sink nodes
/// \ingroup ov_ops_cpp_api
class OPENVINO_API Sink : public Op {
public:
    ~Sink() override = 0;
    OPENVINO_OP("Sink", "util", Op);

protected:
    Sink() : Op() {}

    explicit Sink(const OutputVector& arguments) : Op(arguments) {}
};
}  // namespace op
using SinkVector = std::vector<std::shared_ptr<op::Sink>>;
}  // namespace ov
