// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <list>
#include <memory>
#include <typeinfo>
#include <vector>

#include "openvino/pass/pass.hpp"
#include "openvino/pass/validate.hpp"

namespace ov {
namespace pass {
/**
 * @brief Manager class allows to manage transformation passes
 * @ingroup ov_pass_cpp_api
 */
class OPENVINO_API Manager {
public:
    Manager();
    virtual ~Manager();

    //// \brief Construct Manager with a provided name.
    explicit Manager(std::string name);

    //// \brief Construct Manager with shared PassConfig instance
    explicit Manager(std::shared_ptr<PassConfig> pass_config, std::string name = "UnnamedManager");

    /// \brief Register given transformation class type to execution list
    /// Example below show the basic usage of pass::Manager
    ///
    ///     pass::Manager manager;
    ///     manager.register_pass<MyTransformation>(/* transformation constructor args */);
    ///     manager.run_passes(f);
    ///
    /// For some purposes transformation can be registered and disabled by default.
    ///
    ///     manager.register_pass<MyTransformation, false>();
    ///
    /// \return shared_ptr to the transformation instance
    template <typename T, bool Enable = true, class... Args>
    std::shared_ptr<T> register_pass(Args&&... args) {
        auto rc = push_pass<T>(std::forward<Args>(args)...);
        rc->set_pass_config(m_pass_config);
        if (m_per_pass_validation) {
            push_pass<Validate>();
        }
        if (!Enable && !m_pass_config->is_enabled<T>()) {
            m_pass_config->disable<T>();
        }
        return rc;
    }

    std::shared_ptr<PassBase> register_pass_instance(std::shared_ptr<PassBase> pass) {
        pass->set_pass_config(m_pass_config);
        m_pass_list.push_back(pass);
        if (m_per_pass_validation) {
            push_pass<Validate>();
        }
        return pass;
    }

    /// \brief      Runs registered transformations on a given model
    ///
    /// \param      model Input model
    ///
    /// \return     Returns true if the model was changed by transformations,
    ///             false otherwise.
    bool run_passes(const std::shared_ptr<Model>& model);

    /// \brief Set flag to enable/disable running Validate pass after executing
    /// each registered pass
    /// \param new_state Value "true" enables Validate pass run; "false", otherwise
    void set_per_pass_validation(bool new_state);

    /// \return PassConfig shared object. This object is used for transformations pipeline
    /// configuration.
    /// This object allows to disable/enable transformations execution, set callback to
    /// particular
    /// transformation. For more details see PassConfig class.
    std::shared_ptr<PassConfig> get_pass_config() {
        return m_pass_config;
    }

protected:
    template <typename T, class... Args>
    std::shared_ptr<T> push_pass(Args&&... args) {
        static_assert(std::is_base_of<pass::PassBase, T>::value, "pass not derived from pass base");
        auto pass = std::make_shared<T>(std::forward<Args>(args)...);
        auto pass_base = std::static_pointer_cast<PassBase>(pass);
        m_pass_list.push_back(pass_base);
        return pass;
    }

    std::shared_ptr<PassConfig> m_pass_config;
    std::vector<std::shared_ptr<PassBase>> m_pass_list;
    bool m_per_pass_validation = true;
    std::string m_name = "UnnamedManager";

private:
    bool run_pass(const std::shared_ptr<PassBase>& pass, const std::shared_ptr<Model>& model, bool needs_validate);
};
}  // namespace pass
}  // namespace ov
