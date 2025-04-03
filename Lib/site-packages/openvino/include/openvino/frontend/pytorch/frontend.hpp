// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/frontend/extension/conversion.hpp"
#include "openvino/frontend/extension/telemetry.hpp"
#include "openvino/frontend/frontend.hpp"
#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/frontend/pytorch/visibility.hpp"

namespace ov {
namespace frontend {
namespace pytorch {

class PYTORCH_FRONTEND_API FrontEnd : public ov::frontend::FrontEnd {
public:
    using Ptr = std::shared_ptr<FrontEnd>;
    FrontEnd();

    /// \brief Completely convert and normalize entire Model, throws if it is not possible
    /// \param model Input model
    /// \return fully converted OV Model
    std::shared_ptr<Model> convert(const ov::frontend::InputModel::Ptr& model) const override;

    /// \brief Completely convert the remaining, not converted part of a Model.
    /// \param partiallyConverted partially converted OV Model
    void convert(const std::shared_ptr<Model>& partiallyConverted) const override;

    /// \brief Convert only those parts of the model that can be converted leaving others
    /// as-is. Converted parts are not normalized by additional transformations; normalize
    /// function or another form of convert function should be called to finalize the
    /// conversion process.
    /// \param model Input model
    /// \return partially converted OV Model
    std::shared_ptr<Model> convert_partially(const InputModel::Ptr& model) const override;

    /// \brief Convert operations with one-to-one mapping with decoding nodes.
    /// Each decoding node is an OV node representing a single FW operation node with
    /// all attributes represented in FW-independent way.
    /// \param model Input model
    /// \return OV Model after decoding
    std::shared_ptr<Model> decode(const InputModel::Ptr& model) const override;

    /// \brief Runs normalization passes on Model that was loaded with partial conversion
    /// \param Model partially converted OV Model
    void normalize(const std::shared_ptr<ov::Model>& model) const override;

    /// \brief Gets name of this FrontEnd. Can be used by clients
    /// if frontend is selected automatically by FrontEndManager::load_by_model
    /// \return Paddle frontend name.
    std::string get_name() const override {
        return "pytorch";
    }

    /// \brief Register base extension in the FrontEnd
    /// \param extension base extension
    void add_extension(const std::shared_ptr<ov::Extension>& extension) override;

protected:
    bool supported_impl(const std::vector<ov::Any>& variants) const override;
    ov::frontend::InputModel::Ptr load_impl(const std::vector<ov::Any>& variants) const override;
    std::unordered_map<std::string, CreatorFunction> get_supported_ops(
        const ov::frontend::InputModel::Ptr& model) const;

    std::map<std::string, CreatorFunction> m_op_extension_translators;
    std::vector<ConversionExtensionBase::Ptr> m_conversion_extensions;
    TelemetryExtension::Ptr m_telemetry;
};

}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
