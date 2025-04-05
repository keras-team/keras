// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/descriptor/tensor.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/runtime_attribute.hpp"
#include "openvino/core/visibility.hpp"

namespace ov {

OPENVINO_API void skip_invalidation(const ov::Output<ov::Node>& output);

OPENVINO_API bool skip_invalidation(const ov::descriptor::Tensor& tensor);

OPENVINO_API void remove_skip_invalidation_rti(const std::shared_ptr<ov::Model>& model, bool outermost_model = true);

OPENVINO_API void populate_tensor_with_missing_symbols(ov::descriptor::Tensor& tensor);

/**
 * @ingroup ov_runtime_attr_api
 * @brief SkipInvalidation class represents runtime info attribute that instructs ov::Output objects to skip
 * invalidation of partial values and symbols during partial value propagation.
 */
class OPENVINO_API SkipInvalidation : public RuntimeAttribute {
public:
    OPENVINO_RTTI("SkipInvalidation", "0", RuntimeAttribute);
    SkipInvalidation() = default;
    bool is_copyable() const override {
        return false;
    }
};

}  // namespace ov
