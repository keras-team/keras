// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief a header that defines wrappers for internal NPU plugin-specific
 * LevelZero context and LevelZero shared memory tensors
 *
 * @file openvino/runtime/intel_npu/level_zero/level_zero.hpp
 */
#pragma once

#include "openvino/runtime/core.hpp"
#include "openvino/runtime/intel_npu/properties.hpp"
#include "openvino/runtime/intel_npu/remote_properties.hpp"
#include "openvino/runtime/remote_context.hpp"
#include "openvino/runtime/remote_tensor.hpp"

namespace ov {
namespace intel_npu {

/**
 * @defgroup ov_runtime_level_zero_npu_cpp_api Intel NPU LevelZero interoperability
 * @ingroup ov_runtime_cpp_api
 * Set of C++ classes and properties to work with Remote API for Intel NPU LevelZero plugin.
 */

/**
 * @brief Namespace with Intel NPU LevelZero specific remote objects
 */
namespace level_zero {

/**
 * @brief This class represents an abstraction for NPU plugin remote tensor
 * which can be shared with user-supplied LevelZero buffer.
 * The plugin object derived from this class can be obtained with ZeroContext::create_tensor() call.
 * @note User can obtain Level Zero buffer handle from this class.
 * @ingroup ov_runtime_level_zero_npu_cpp_api
 */
class ZeroBufferTensor : public RemoteTensor {
public:
    /**
     * @brief Checks that type defined runtime parameters are presented in remote object
     * @param tensor a tensor to check
     */
    static void type_check(const Tensor& tensor) {
        RemoteTensor::type_check(
            tensor,
            {{std::string(mem_handle.name()), {}},
             {std::string(mem_type.name()),
              {ov::Any(MemType::L0_INTERNAL_BUF).as<std::string>(), ov::Any(MemType::SHARED_BUF).as<std::string>()}}});
    }

    /**
     * @brief Returns the underlying LevelZero memory object handle.
     * @return underlying void* memory object handle
     */
    void* get() {
        return get_params().at(mem_handle.name()).as<void*>();
    }
};

/**
 * @brief This class represents an abstraction for NPU plugin remote context
 * which is shared with LevelZero context object.
 * The plugin object derived from this class can be obtained either with
 * CompiledModel::get_context() or Core::create_context() calls.
 * @ingroup ov_runtime_level_zero_npu_cpp_api
 */
class ZeroContext : public RemoteContext {
protected:
    /**
     * @brief NPU device name
     */
    static constexpr const char* device_name = "NPU";

    /**
     * @brief Default constructor which can be used in derived classes to avoid multiple create_context() calls
     */
    ZeroContext() = default;

public:
    // Needed to make create_tensor overloads from base class visible for user
    using RemoteContext::create_tensor;

    /**
     * @brief Constructs context object from user-supplied LevelZero context handle
     * @param core A reference to OpenVINO Runtime Core object
     */
    ZeroContext(Core& core) {
        *this = core.get_default_context(device_name).as<ZeroContext>();
    }

    /**
     * @brief Returns the underlying LevelZero context handle.
     * @return `void*`
     */
    void* get() {
        return get_params().at(l0_context.name()).as<void*>();
    }

    /**
     * @brief This function is used to obtain remote tensor object from user-supplied NT handle object
     * @param type Tensor element type
     * @param shape Tensor shape
     * @param buffer A void* object that should be wrapped by a remote tensor
     * @return A remote tensor instance
     */
    ZeroBufferTensor create_tensor(const element::Type type, const Shape& shape, void* buffer) {
        AnyMap params = {{mem_type.name(), MemType::SHARED_BUF}, {mem_handle.name(), buffer}};
        return create_tensor(type, shape, params).as<ZeroBufferTensor>();
    }

    /**
     * @brief This function is used to obtain remote tensor object from user-supplied DMA-BUF System Heap object
     * @param type Tensor element type
     * @param shape Tensor shape
     * @param fd A int object that should be wrapped by a remote tensor
     * @return A remote tensor instance
     */
    ZeroBufferTensor create_tensor(const element::Type type, const Shape& shape, int fd) {
        AnyMap params = {{mem_type.name(), MemType::SHARED_BUF},
                         {mem_handle.name(), reinterpret_cast<void*>(static_cast<intptr_t>(fd))}};
        return create_tensor(type, shape, params).as<ZeroBufferTensor>();
    }

    /**
     * @brief This function is used to obtain remote tensor object
     * @param type Tensor element type
     * @param shape Tensor shape
     * @param tensor_type Type of the tensor to be shared, input, output or binded
     * @return A remote tensor instance
     */
    ZeroBufferTensor create_l0_host_tensor(const element::Type type,
                                           const Shape& shape,
                                           const TensorType tensor_type = TensorType::BINDED) {
        AnyMap params = {{mem_type.name(), MemType::L0_INTERNAL_BUF}, {ov::intel_npu::tensor_type.name(), tensor_type}};
        return create_tensor(type, shape, params).as<ZeroBufferTensor>();
    }
};

}  // namespace level_zero
}  // namespace intel_npu
}  // namespace ov
