// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/frontend/extension/conversion.hpp"
#include "openvino/frontend/frontend.hpp"
#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/frontend/pytorch/visibility.hpp"

namespace ov {
namespace frontend {
namespace pytorch {

class PYTORCH_FRONTEND_API ConversionExtension : public ConversionExtensionBase {
public:
    using Ptr = std::shared_ptr<ConversionExtension>;

    ConversionExtension() = delete;

    ConversionExtension(const std::string& op_type, const ov::frontend::CreatorFunction& converter)
        : ConversionExtensionBase(op_type),
          m_converter(converter) {}

    const ov::frontend::CreatorFunction& get_converter() const {
        return m_converter;
    }

    ~ConversionExtension() override;

private:
    ov::frontend::CreatorFunction m_converter;
};

}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
