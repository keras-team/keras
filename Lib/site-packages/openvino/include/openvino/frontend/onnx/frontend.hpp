// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/frontend/extension/conversion.hpp"
#include "openvino/frontend/extension/decoder_transformation.hpp"
#include "openvino/frontend/extension/holder.hpp"
#include "openvino/frontend/extension/telemetry.hpp"
#include "openvino/frontend/frontend.hpp"
#include "openvino/frontend/onnx/visibility.hpp"

namespace ov {
namespace frontend {
namespace onnx {

class ONNX_FRONTEND_API FrontEnd : public ov::frontend::FrontEnd {
public:
    using Ptr = std::shared_ptr<FrontEnd>;
    std::shared_ptr<ov::Model> convert(const InputModel::Ptr& model) const override;
    void convert(const std::shared_ptr<ov::Model>& partially_converted) const override;
    std::shared_ptr<ov::Model> convert_partially(const InputModel::Ptr& input_model) const override;
    std::shared_ptr<ov::Model> decode(const InputModel::Ptr& input_model) const override;
    std::string get_name() const override;
    bool supported_impl(const std::vector<ov::Any>& variants) const override;
    void add_extension(const std::shared_ptr<ov::Extension>& extension) override;
    void normalize(const std::shared_ptr<ov::Model>& model) const override;

protected:
    InputModel::Ptr load_impl(const std::vector<ov::Any>& params) const override;

    // m_other_extensions should be the first member here,
    // m_other_extensions can contain SO Extension (holder for other Extensions),
    // so it should be released last.
    std::vector<Extension::Ptr> m_other_extensions;
    std::vector<DecoderTransformationExtension::Ptr> m_transformation_extensions;
    ExtensionHolder m_extensions;
    std::once_flag has_legacy_extension;
};

}  // namespace onnx
}  // namespace frontend
}  // namespace ov
