// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/core_visibility.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/runtime_attribute.hpp"

namespace ov {

void OPENVINO_API mark_as_precision_sensitive(ov::Input<ov::Node> node_input);

void OPENVINO_API unmark_as_precision_sensitive(ov::Input<ov::Node> node_input);

bool OPENVINO_API is_precision_sensitive(const ov::Input<ov::Node>& node_input);

/**
 * @brief PrecisionSensitive class represents runtime info attribute that marks
 * input to an operation as a precision sensitive and disables compression to FP16
 * of the subgraph before this input.
 */
class OPENVINO_API PrecisionSensitive : public RuntimeAttribute {
public:
    OPENVINO_RTTI("precision_sensitive", "0", RuntimeAttribute);

    PrecisionSensitive() = default;

    bool is_copyable() const override {
        return false;
    }
};

}  // namespace ov
