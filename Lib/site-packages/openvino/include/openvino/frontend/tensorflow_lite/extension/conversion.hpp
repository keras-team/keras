// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/frontend/extension/conversion.hpp"
#include "openvino/frontend/frontend.hpp"
#include "openvino/frontend/tensorflow_lite/visibility.hpp"

namespace ov {
namespace frontend {
namespace tensorflow_lite {

class TENSORFLOW_LITE_FRONTEND_API ConversionExtension : public ConversionExtensionBase {
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

}  // namespace tensorflow_lite
}  // namespace frontend
}  // namespace ov
