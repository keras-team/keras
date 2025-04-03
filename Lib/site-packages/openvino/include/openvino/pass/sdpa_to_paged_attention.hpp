// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include "openvino/pass/pass.hpp"

namespace ov {
namespace pass {
/**
 * @brief The transformation replaces KV-cache processing part in LLMs by PagedAttention operation.
 * \ingroup ov_pass_cpp_api
 */
class OPENVINO_API SDPAToPagedAttention : public ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("SDPAToPagedAttention");

    explicit SDPAToPagedAttention(bool use_per_layer_block_indices_inputs = false,
                                  bool use_score_outputs = false,
                                  bool allow_cache_rotation = false);
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;

private:
    bool m_use_per_layer_block_indices_inputs;
    bool m_use_score_outputs;
    bool m_allow_cache_rotation;
};
}  // namespace pass
}  // namespace ov
