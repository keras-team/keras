// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/frontend/extension/conversion.hpp"
#include "openvino/frontend/frontend.hpp"
#include "openvino/frontend/paddle/node_context.hpp"
#include "openvino/frontend/paddle/visibility.hpp"

namespace ov {
namespace frontend {
namespace paddle {

class PADDLE_FRONTEND_API ConversionExtension : public ConversionExtensionBase {
public:
    using Ptr = std::shared_ptr<ConversionExtension>;

    ConversionExtension() = delete;

    ConversionExtension(const std::string& op_type, const ov::frontend::CreatorFunctionNamed& converter)
        : ConversionExtensionBase(op_type),
          m_converter(converter) {}

    ~ConversionExtension() override;

    const ov::frontend::CreatorFunctionNamed& get_converter() const {
        return m_converter;
    }

private:
    ov::frontend::CreatorFunctionNamed m_converter;
};

}  // namespace paddle
}  // namespace frontend
}  // namespace ov
