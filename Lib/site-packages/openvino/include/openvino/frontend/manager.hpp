// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <sstream>
#include <string>

#include "openvino/core/any.hpp"
#include "openvino/frontend/frontend.hpp"
#include "openvino/frontend/visibility.hpp"
#include "openvino/runtime/common.hpp"

namespace ov {
// Forward declaration
OPENVINO_RUNTIME_API void shutdown();
namespace frontend {
// -------------- FrontEndManager -----------------
using FrontEndFactory = std::function<FrontEnd::Ptr()>;

/// \brief Frontend management class, loads available frontend plugins on construction
/// Allows load of frontends for particular framework, register new and list available
/// frontends This is a main frontend entry point for client applications
class FRONTEND_API FrontEndManager final {
public:
    /// \brief Default constructor. Searches and loads of available frontends
    FrontEndManager();

    /// \brief Default move constructor
    FrontEndManager(FrontEndManager&&) noexcept;

    /// \brief Default move assignment operator
    FrontEndManager& operator=(FrontEndManager&&) noexcept;

    /// \brief Default destructor
    ~FrontEndManager();

    /// \brief Loads frontend by name of framework and capabilities
    ///
    /// \param framework Framework name. Throws exception if name is not in list of
    /// available frontends
    ///
    /// \return Frontend interface for further loading of models
    FrontEnd::Ptr load_by_framework(const std::string& framework);

    /// \brief Loads frontend by model fragments described by each FrontEnd documentation.
    /// Selects and loads appropriate frontend depending on model file extension and other
    /// file info (header)
    ///
    /// \param vars Any number of parameters of any type. What kind of parameters
    /// are accepted is determined by each FrontEnd individually, typically it is
    /// std::string containing path to the model file. For more information please
    /// refer to specific FrontEnd documentation.
    ///
    /// \return Frontend interface for further loading of model. Returns 'nullptr'
    /// if no suitable frontend is found
    template <typename... Types>
    FrontEnd::Ptr load_by_model(const Types&... vars) {
        return load_by_model_impl({ov::Any{vars}...});
    }
    FrontEnd::Ptr load_by_model(const std::vector<ov::Any>& variants) {
        return load_by_model_impl(variants);
    }

    /// \brief Gets list of registered frontends. Any not loaded frontends will be loaded by this call
    std::vector<std::string> get_available_front_ends();

    /// \brief Register frontend with name and factory creation method
    ///
    /// \param name Name of front end
    ///
    /// \param creator Creation factory callback. Will be called when frontend is about to
    /// be created
    void register_front_end(const std::string& name, FrontEndFactory creator);

    /// \brief Register frontend with name and factory loaded from provided library
    ///
    /// \param name Name of front end
    ///
    /// \param library_path Path (absolute or relative) or name of a frontend library. If name is
    /// provided, depending on platform, it will be wrapped with shared library suffix and prefix
    /// to identify library full name
    void register_front_end(const std::string& name, const std::string& library_path);

private:
    class Impl;

    FrontEnd::Ptr load_by_model_impl(const std::vector<ov::Any>& variants);

    std::unique_ptr<Impl> m_impl;

    friend OPENVINO_RUNTIME_API void ov::shutdown();
    /// \brief Shutdown the manager by try releasing frontend libraries
    static void shutdown();
};

template <>
FRONTEND_API FrontEnd::Ptr FrontEndManager::load_by_model(const std::vector<ov::Any>& variants);

// --------- Plugin exporting information --------------

/// \brief Each frontend plugin is responsible to export get_api_version function returning
/// version of frontend API used for this plugin
/// If version is not matched with OV_FRONTEND_API_VERSION - plugin will not be loaded by
/// FrontEndManager
using FrontEndVersion = uint64_t;

/// \brief Each frontend plugin is responsible to export get_front_end_data function returning
/// heap-allocated pointer to this structure. Will be used by FrontEndManager during loading
/// of plugins
struct FrontEndPluginInfo {
    std::string m_name;
    FrontEndFactory m_creator;
};

}  // namespace frontend
}  // namespace ov
