// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header file for the OpenVINO Runtime RemoteContext class.
 * @file openvino/runtime/remote_context.hpp
 */
#pragma once

#include <map>
#include <memory>
#include <string>

#include "openvino/core/shape.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/runtime/common.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/runtime/remote_tensor.hpp"

namespace ov {

class Core;
class IRemoteContext;
class CompiledModel;

/**
 * @brief This class represents an abstraction
 * @ingroup ov_runtime_cpp_api
 * for remote (non-CPU) accelerator device-specific inference context.
 * Such context represents a scope on the device within which compiled
 * models and remote memory tensors can exist, function, and exchange data.
 */
class OPENVINO_RUNTIME_API RemoteContext {
protected:
    std::shared_ptr<IRemoteContext> _impl;  //!< Pointer to the remote context implementation.
    std::shared_ptr<void> _so;              //!< Reference to the shared object that loaded implementation.

    /**
     * @brief Constructs RemoteContext from the initialized std::shared_ptr.
     * @param impl Initialized shared pointer.
     * @param so Plugin to use. This is required to ensure that RemoteContext can work properly even if a plugin
     * object is destroyed.
     */
    RemoteContext(const std::shared_ptr<IRemoteContext>& impl, const std::shared_ptr<void>& so);
    friend class ov::Core;
    friend class ov::CompiledModel;

public:
    /**
     * @brief Default constructor.
     */
    RemoteContext() = default;

    /**
     * @brief Default copy constructor.
     * @param other Another RemoteContext object.
     */
    RemoteContext(const RemoteContext& other) = default;

    /**
     * @brief Default copy assignment operator.
     * @param other Another RemoteContext object.
     * @return Reference to the current object.
     */
    RemoteContext& operator=(const RemoteContext& other) = default;

    /**
     * @brief Default move constructor.
     * @param other Another RemoteContext object.
     */
    RemoteContext(RemoteContext&& other) = default;

    /**
     * @brief Default move assignment operator.
     * @param other Another RemoteContext object.
     * @return Reference to the current object.
     */
    RemoteContext& operator=(RemoteContext&& other) = default;

    /**
     * @brief Checks if current RemoteContext object is initialized
     * @return `true` if current RemoteContext object is initialized, `false` - otherwise
     */
    operator bool() const noexcept;

    /**
     * @brief Destructor that preserves unloading order of implementation object and reference to the library.
     */
    ~RemoteContext();

    /**
     * @brief Internal method: checks remote type.
     * @param remote_context Remote context which type is checked.
     * @param type_info Map with remote object runtime info.
     * @throw Exception if type check with the specified parameters failed.
     */
    static void type_check(const RemoteContext& remote_context,
                           const std::map<std::string, std::vector<std::string>>& type_info = {});

    /**
     * @brief Checks if the RemoteContext object can be cast to the type T.
     *
     * @tparam T Type to be checked. Must represent a class derived from RemoteContext.
     * @return True if this object can be dynamically cast to the type T*; false, otherwise.
     */
    template <typename T>
    bool is() const noexcept {
        static_assert(std::is_base_of<RemoteContext, T>::value,
                      "Could not check type that is not inherited from RemoteContext");
        try {
            T::type_check(*this);
        } catch (...) {
            return false;
        }
        return true;
    }

    /**
     * @brief Casts this RemoteContext object to the type T.
     *
     * @tparam T Type to cast to. Must represent a class derived from RemoteContext.
     * @return T Object.
     */
    template <typename T>
    const T as() const {
        static_assert(std::is_base_of<RemoteContext, T>::value,
                      "Could not check type that is not inherited from RemoteContext");
        T::type_check(*this);
        return *static_cast<const T*>(this);
    }

    /**
     * @brief Returns name of a device on which underlying object is allocated.
     * Abstract method.
     * @return A device name string in fully specified format `<device_name>[.<device_id>[.<tile_id>]]` (e.g. GPU.0.1).
     */
    std::string get_device_name() const;

    /**
     * @brief Allocates memory tensor in device memory or wraps user-supplied memory handle
     * using the specified tensor description and low-level device-specific parameters.
     * Returns a pointer to the object that implements the RemoteTensor interface.
     * @param type Defines the element type of the tensor.
     * @param shape Defines the shape of the tensor.
     * @param params Map of the low-level tensor object parameters.
     * @return Pointer to a plugin object that implements the RemoteTensor interface.
     */
    RemoteTensor create_tensor(const element::Type& type, const Shape& shape, const AnyMap& params = {});

    /**
     * @brief Returns a map of device-specific parameters required for low-level
     * operations with the underlying object.
     * Parameters include device/context handles, access flags,
     * etc. Content of the returned map depends on a remote execution context that is
     * currently set on the device (working scenario).
     * Abstract method.
     * @return A map of name/parameter elements.
     */
    AnyMap get_params() const;

    /**
     * @brief This method is used to create a host tensor object friendly for the device in current context.
     * For example, GPU context may allocate USM host memory (if corresponding extension is available),
     * which could be more efficient than regular host memory.
     * @param type Tensor element type.
     * @param shape Tensor shape.
     * @return A tensor instance with device friendly memory.
     */
    Tensor create_host_tensor(const element::Type type, const Shape& shape);
};

}  // namespace ov
