// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"

namespace ov {
namespace pass {
/**
 * @brief ConvertFP32ToFP16 transformation
 * @ingroup ov_pass_cpp_api
 */
class OPENVINO_API ConvertFP32ToFP16 : public ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("ConvertFP32ToFP16");
    bool run_on_model(const std::shared_ptr<ov::Model>&) override;
};
}  // namespace pass
}  // namespace ov
