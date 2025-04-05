// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "openvino/core/any.hpp"
#include "openvino/core/extension.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/op_extension.hpp"
#include "openvino/frontend/input_model.hpp"
#include "openvino/frontend/visibility.hpp"

#ifdef OPENVINO_CPP_VER_AT_LEAST_17
#    include <filesystem>
#endif

namespace ov {
namespace frontend {
/// \brief An interface for identifying a frontend for a particular framework.
/// Provides an ability to load and convert of input model
class FRONTEND_API FrontEnd {
    friend class FrontEndManager;

    std::shared_ptr<void> m_shared_object = {};  // Library handle
    std::shared_ptr<FrontEnd> m_actual = {};

public:
    using Ptr = std::shared_ptr<FrontEnd>;

    /// \brief Default constructor
    FrontEnd();

    FrontEnd(const FrontEnd&) = delete;

    FrontEnd(FrontEnd&&) = delete;

    FrontEnd& operator=(const FrontEnd&) = delete;

    FrontEnd& operator=(FrontEnd&&) = delete;

    virtual ~FrontEnd();

    /// \brief Validates if FrontEnd can recognize model with parameters specified.
    /// Same parameters should be used to load model.
    /// \param vars Any number of parameters of any type. What kind of parameters
    /// are accepted is determined by each FrontEnd individually, typically it is
    /// std::string containing path to the model file. For more information please
    /// refer to specific FrontEnd documentation.
    /// \return true if model recognized, false - otherwise.
    template <typename... Types>
    inline bool supported(const Types&... vars) const {
#ifdef OPENVINO_CPP_VER_AT_LEAST_17
        if constexpr ((std::is_same_v<std::filesystem::path, Types> || ...)) {
            return supported_impl({path_as_str_or_forward(vars)...});
        } else
#endif
            return supported_impl({ov::Any(vars)...});
    }
    inline bool supported(const ov::AnyVector& vars) const {
        return supported_impl(vars);
    }

    /// \brief Loads an input model by any specified arguments. Each FrontEnd separately
    /// defines what arguments it can accept.
    /// \param vars Any number of parameters of any type. What kind of parameters
    /// are accepted is determined by each FrontEnd individually, typically it is
    /// std::string containing path to the model file. For more information please
    /// refer to specific FrontEnd documentation.
    /// \return Loaded input model.
    template <typename... Types>
    inline InputModel::Ptr load(const Types&... vars) const {
#ifdef OPENVINO_CPP_VER_AT_LEAST_17
        if constexpr ((std::is_same_v<std::filesystem::path, Types> || ...)) {
            return load_impl({path_as_str_or_forward(vars)...});
        } else
#endif
            return load_impl({ov::Any{vars}...});
    }

    inline InputModel::Ptr load(const ov::AnyVector& vars) const {
        return load_impl(vars);
    }

    /// \brief Completely convert and normalize entire Model, throws if it is not
    /// possible
    /// \param model Input model
    /// \return fully converted OV Model
    virtual std::shared_ptr<ov::Model> convert(const InputModel::Ptr& model) const;

    /// \brief Completely convert the remaining, not converted part of a Model.
    /// \param partiallyConverted partially converted OV Model
    virtual void convert(const std::shared_ptr<ov::Model>& partially_converted) const;

    /// \brief Convert only those parts of the model that can be converted leaving others
    /// as-is wrapped by FrameworkNode. Converted parts are normalized by additional
    /// transformations like it is done in convert method. If part of the graph cannot be
    /// converted, it is not guaranteed that the converted regions are completely normalized.
    /// Normalize should be called for each completely converted parts individually in this case.
    /// \param model Input model
    /// \return partially converted OV Model
    virtual std::shared_ptr<ov::Model> convert_partially(const InputModel::Ptr& model) const;

    /// \brief Convert operations with one-to-one mapping with decoding nodes.
    /// Each decoding node is an OV node representing a single FW operation node with
    /// all attributes represented in FW-independent way.
    /// \param model Input model
    /// \return OV Model after decoding
    virtual std::shared_ptr<ov::Model> decode(const InputModel::Ptr& model) const;

    /// \brief Runs normalization passes on Model that was loaded with partial conversion
    /// \param Model partially converted OV Model
    virtual void normalize(const std::shared_ptr<ov::Model>& model) const;

    /// \brief Gets name of this FrontEnd. Can be used by clients
    /// if frontend is selected automatically by FrontEndManager::load_by_model
    ///
    /// \return Current frontend name. Empty string if not implemented
    virtual std::string get_name() const;

    /// \brief Register base extension in the FrontEnd
    /// \param extension base extension
    virtual void add_extension(const std::shared_ptr<ov::Extension>& extension);

    /// \brief Register base extensions in the FrontEnd
    /// \param extensions vector of extensions
    void add_extension(const std::vector<std::shared_ptr<ov::Extension>>& extensions);

    /// \brief Registers extension
    /// \param library_path path to library with ov::Extension
    /// \{
    void add_extension(const std::string& library_path);

#ifdef OPENVINO_CPP_VER_AT_LEAST_17
    void add_extension(const std::filesystem::path& library_path) {
        add_extension(library_path.string());
    }
#endif
    /// \}

#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT

    /// \brief Registers extension
    /// \param library_path path to library with ov::Extension
    void add_extension(const std::wstring& library_path);

#endif

    /// @brief Registers extension
    /// @param extension Extension class which is inherited from ov::BaseOpExtension class
    template <class T, typename std::enable_if<std::is_base_of<ov::Extension, T>::value, bool>::type = true>
    void add_extension(const T& extension) {
        std::shared_ptr<ov::Extension> ext = std::make_shared<T>(extension);
        add_extension(ext);
    }

    /// @brief Registers extensions
    /// @param extension Extension class which is inherited from ov::Extension class
    template <class T,
              class... Targs,
              typename std::enable_if<std::is_base_of<ov::Extension, T>::value, bool>::type = true>
    void add_extension(const T& extension, Targs... args) {
        std::shared_ptr<ov::Extension> ext = std::make_shared<T>(extension);
        add_extension(ext);
        add_extension(args...);
    }

protected:
    virtual bool supported_impl(const std::vector<ov::Any>& variants) const;

    virtual InputModel::Ptr load_impl(const std::vector<ov::Any>& variants) const;

    void validate_path(const std::string& path) const;
#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
    void validate_path(const std::wstring& path) const;
#endif

    std::vector<ov::Extension::Ptr> m_extensions;

private:
    static std::shared_ptr<ov::Model> create_copy(const std::shared_ptr<ov::Model>& ov_model,
                                                  const std::shared_ptr<void>& shared_object);

#ifdef OPENVINO_CPP_VER_AT_LEAST_17
    template <class T>
    static constexpr auto path_as_str_or_forward(T&& p) {
        if constexpr (std::is_same_v<std::filesystem::path, std::decay_t<T>>) {
            return p.string();
        } else {
            return std::forward<T>(p);
        }
    }
#endif
};

template <>
inline bool FrontEnd::supported(const std::vector<ov::Any>& variants) const {
    return supported_impl(variants);
}

}  // namespace frontend

}  // namespace ov
