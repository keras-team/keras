// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/pass.hpp"

namespace ov {
namespace pass {
/**
 * @brief The transformation converts KV cache state back to stateless form.
 * \ingroup ov_pass_cpp_api
 */
class OPENVINO_API StatefulToStateless : public ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("StatefulToStateless");

    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};
}  // namespace pass
}  // namespace ov
