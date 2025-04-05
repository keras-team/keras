// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "openvino/frontend/extension/conversion.hpp"
#include "openvino/frontend/node_context.hpp"
#include "openvino/frontend/onnx/node_context.hpp"
#include "openvino/frontend/onnx/visibility.hpp"

namespace ov {
namespace frontend {
namespace onnx {
class ONNX_FRONTEND_API ConversionExtension : public ConversionExtensionBase {
public:
    using Ptr = std::shared_ptr<ConversionExtension>;

    ConversionExtension(const std::string& op_type, const ov::frontend::CreatorFunction& converter)
        : ConversionExtensionBase(op_type),
          m_converter(converter) {}

    ConversionExtension(const std::string& op_type,
                        const std::string& domain,
                        const ov::frontend::CreatorFunction& converter)
        : ConversionExtensionBase(op_type),
          m_domain{domain},
          m_converter(converter) {}

    ~ConversionExtension() override;

    const std::string& get_domain() const {
        return m_domain;
    }

    const ov::frontend::CreatorFunction& get_converter() const {
        return m_converter;
    }

private:
    std::string m_domain = "";
    ov::frontend::CreatorFunction m_converter;
};
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
