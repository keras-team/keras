// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <fstream>
#include <iostream>
#include <type_traits>

#include "openvino/core/extension.hpp"
#include "openvino/frontend/visibility.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/pass.hpp"

namespace ov {
namespace frontend {

/// \brief Holds a transformation that is applied just after the original model graph is decoded.
/// This class is a holder for transformation. The transformation can be specified as
/// FunctionPass or MathcerPass derivatives or as a function that can be used to build corresponding
/// FunctionPass or MatcherPass object. The type of the extension is determined in the moment of creation by
/// calling corresponding ctor.
class FRONTEND_API DecoderTransformationExtension : public ov::Extension {
public:
    using Ptr = std::shared_ptr<DecoderTransformationExtension>;
    DecoderTransformationExtension() = default;

    /// \brief Create a custom functional pass where code of the pass is implemented as a function.
    explicit DecoderTransformationExtension(const std::function<bool(std::shared_ptr<ov::Model>)>& function_pass);

    /// \brief Create a custom matcher pass where the code of matcher pass initialization is a given function.
    explicit DecoderTransformationExtension(
        const std::function<void(ov::pass::MatcherPass*)>& matcher_pass_initializer);

    /// \brief Register existing transformation object which will be copied and kept for further registration.
    template <typename Transformation,
              typename std::enable_if<std::is_base_of<ov::pass::PassBase, Transformation>::value, bool>::type = true>
    explicit DecoderTransformationExtension(const Transformation& transformation)
        : m_registration([=](ov::pass::Manager& manager) {
              manager.register_pass<Transformation>(transformation);
          }) {}

    /// \brief Register pass from this object in a given pass manager object
    void register_pass(ov::pass::Manager& manager) const;

protected:
    void set_registration(const std::function<void(ov::pass::Manager&)>& registration) {
        m_registration = registration;
    }

private:
    std::function<void(ov::pass::Manager&)> m_registration;
};
}  // namespace frontend
}  // namespace ov
