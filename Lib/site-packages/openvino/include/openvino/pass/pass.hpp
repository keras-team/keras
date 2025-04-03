// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>

#include "openvino/core/core_visibility.hpp"
#include "openvino/core/enum_mask.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/rtti.hpp"
#include "openvino/pass/pass_config.hpp"

#define _OPENVINO_MODEL_PASS_RTTI_WITH_TYPE(TYPE_NAME) _OPENVINO_MODEL_PASS_RTTI_WITH_TYPE_VERSION(TYPE_NAME, "0")

#define _OPENVINO_MODEL_PASS_RTTI_WITH_TYPE_VERSION(TYPE_NAME, VERSION_NAME) \
    _OPENVINO_RTTI_WITH_TYPE_VERSION_PARENT(TYPE_NAME, VERSION_NAME, ::ov::pass::ModelPass)

#define OPENVINO_MODEL_PASS_RTTI(...)                                                                       \
    _OPENVINO_RTTI_EXPAND(_OPENVINO_RTTI_DEFINITION_SELECTOR_2(__VA_ARGS__,                                 \
                                                               _OPENVINO_MODEL_PASS_RTTI_WITH_TYPE_VERSION, \
                                                               _OPENVINO_MODEL_PASS_RTTI_WITH_TYPE)(__VA_ARGS__))

namespace ov {
namespace pass {
enum class PassProperty : uint32_t {
    // Pass requires node shapes to be static
    REQUIRE_STATIC_SHAPE = 0x1,
    // Pass transformation will change the function's dynamic state
    CHANGE_DYNAMIC_STATE = 1 << 1,
};

using PassPropertyMask = ov::EnumMask<PassProperty>;

/**
 * @brief Base class for transformation passes
 * @ingroup ov_pass_cpp_api
 */
class OPENVINO_API PassBase {
    friend class Manager;

public:
    PassBase();
    virtual ~PassBase();
    /// Check if this pass has all the pass properties.
    bool get_property(const PassPropertyMask& prop_mask) const;

    void set_name(const std::string& name) {
        m_name = name;
    }
    std::string get_name() const;

    /// \brief Set callback for particular transformation type.
    /// This method set global callback. For more details see PassConfig class
    /// documentation.
    /// \param callback lambda function that takes node and returns bool
    void set_callback(const param_callback& callback);

    /// \brief Set PassConfig for particular transformation instance
    /// \param pass_config is a PassConfig shared_ptr
    virtual void set_pass_config(const std::shared_ptr<PassConfig>& pass_config) {
        m_pass_config = pass_config;
    }

    /// \brief Allows to access PassConfig shared instance
    /// \return Shared instance of PassConfig class
    std::shared_ptr<PassConfig> get_pass_config() {
        return m_pass_config;
    }

    /// \brief Applies callback for given node. By default callback returns false.
    /// \param node which will be used inside callback
    /// \return result of callback execution for given node
    bool transformation_callback(const std::shared_ptr<const Node>& node) {
        return m_pass_config->get_callback(get_type_info())(node);
    }

    using type_info_t = DiscreteTypeInfo;

    virtual const type_info_t& get_type_info() const = 0;

protected:
    void set_property(const PassPropertyMask& prop, bool value);

private:
    PassPropertyMask m_property;

    std::string m_name;
    std::shared_ptr<PassConfig> m_pass_config;
};

/**
 * @brief Base class for Model passes
 * @ingroup ov_pass_cpp_api
 */
class OPENVINO_API ModelPass : public PassBase {
public:
    OPENVINO_RTTI("ov::pass::ModelPass");
    ~ModelPass() override;
    virtual bool run_on_model(const std::shared_ptr<ov::Model>& m) = 0;
};

}  // namespace pass
}  // namespace ov
