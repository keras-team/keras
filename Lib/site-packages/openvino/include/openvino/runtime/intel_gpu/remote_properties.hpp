// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header for properties of shared device contexts and shared device memory blobs for GPU plugin
 *        To use in constructors of Remote objects
 *
 * @file openvino/runtime/intel_gpu/remote_properties.hpp
 */
#pragma once

#include "openvino/runtime/properties.hpp"

namespace ov {
namespace intel_gpu {

using gpu_handle_param = void*;

/**
 * @brief Enum to define the type of the shared context
 * @ingroup ov_runtime_ocl_gpu_cpp_api
 */
enum class ContextType {
    OCL = 0,        //!< Pure OpenCL context
    VA_SHARED = 1,  //!< Context shared with a video decoding device
};

/** @cond INTERNAL */
inline std::ostream& operator<<(std::ostream& os, const ContextType& context_type) {
    switch (context_type) {
    case ContextType::OCL:
        return os << "OCL";
    case ContextType::VA_SHARED:
        return os << "VA_SHARED";
    default:
        OPENVINO_THROW("Unsupported context type");
    }
}

inline std::istream& operator>>(std::istream& is, ContextType& context_type) {
    std::string str;
    is >> str;
    if (str == "OCL") {
        context_type = ContextType::OCL;
    } else if (str == "VA_SHARED") {
        context_type = ContextType::VA_SHARED;
    } else {
        OPENVINO_THROW("Unsupported context type: ", str);
    }
    return is;
}
/** @endcond */

/**
 * @brief Shared device context type: can be either pure OpenCL (OCL)
 * or shared video decoder (VA_SHARED) context
 * @ingroup ov_runtime_ocl_gpu_cpp_api
 */
static constexpr Property<ContextType> context_type{"CONTEXT_TYPE"};

/**
 * @brief This key identifies OpenCL context handle
 * in a shared context or shared memory blob parameter map
 * @ingroup ov_runtime_ocl_gpu_cpp_api
 */
static constexpr Property<gpu_handle_param> ocl_context{"OCL_CONTEXT"};

/**
 * @brief This key identifies ID of device in OpenCL context
 * if multiple devices are present in the context
 * @ingroup ov_runtime_ocl_gpu_cpp_api
 */
static constexpr Property<int> ocl_context_device_id{"OCL_CONTEXT_DEVICE_ID"};

/**
 * @brief In case of multi-tile system,
 * this key identifies tile within given context
 * @ingroup ov_runtime_ocl_gpu_cpp_api
 */
static constexpr Property<int> tile_id{"TILE_ID"};

/**
 * @brief This key identifies OpenCL queue handle in a shared context
 * @ingroup ov_runtime_ocl_gpu_cpp_api
 */
static constexpr Property<gpu_handle_param> ocl_queue{"OCL_QUEUE"};

/**
 * @brief This key identifies video acceleration device/display handle
 * in a shared context or shared memory blob parameter map
 * @ingroup ov_runtime_ocl_gpu_cpp_api
 */
static constexpr Property<gpu_handle_param> va_device{"VA_DEVICE"};

/**
 * @brief Enum to define the type of the shared memory buffer
 * @ingroup ov_runtime_ocl_gpu_cpp_api
 */
enum class SharedMemType {
    OCL_BUFFER = 0,         //!< Shared OpenCL buffer blob
    OCL_IMAGE2D = 1,        //!< Shared OpenCL 2D image blob
    USM_USER_BUFFER = 2,    //!< Shared USM pointer allocated by user
    USM_HOST_BUFFER = 3,    //!< Shared USM pointer type with host allocation type allocated by plugin
    USM_DEVICE_BUFFER = 4,  //!< Shared USM pointer type with device allocation type allocated by plugin
    VA_SURFACE = 5,         //!< Shared video decoder surface or D3D 2D texture blob
    DX_BUFFER = 6           //!< Shared D3D buffer blob
};

/** @cond INTERNAL */
inline std::ostream& operator<<(std::ostream& os, const SharedMemType& share_mem_type) {
    switch (share_mem_type) {
    case SharedMemType::OCL_BUFFER:
        return os << "OCL_BUFFER";
    case SharedMemType::OCL_IMAGE2D:
        return os << "OCL_IMAGE2D";
    case SharedMemType::USM_USER_BUFFER:
        return os << "USM_USER_BUFFER";
    case SharedMemType::USM_HOST_BUFFER:
        return os << "USM_HOST_BUFFER";
    case SharedMemType::USM_DEVICE_BUFFER:
        return os << "USM_DEVICE_BUFFER";
    case SharedMemType::VA_SURFACE:
        return os << "VA_SURFACE";
    case SharedMemType::DX_BUFFER:
        return os << "DX_BUFFER";
    default:
        OPENVINO_THROW("Unsupported memory type");
    }
}

inline std::istream& operator>>(std::istream& is, SharedMemType& share_mem_type) {
    std::string str;
    is >> str;
    if (str == "OCL_BUFFER") {
        share_mem_type = SharedMemType::OCL_BUFFER;
    } else if (str == "OCL_IMAGE2D") {
        share_mem_type = SharedMemType::OCL_IMAGE2D;
    } else if (str == "USM_USER_BUFFER") {
        share_mem_type = SharedMemType::USM_USER_BUFFER;
    } else if (str == "USM_HOST_BUFFER") {
        share_mem_type = SharedMemType::USM_HOST_BUFFER;
    } else if (str == "USM_DEVICE_BUFFER") {
        share_mem_type = SharedMemType::USM_DEVICE_BUFFER;
    } else if (str == "VA_SURFACE") {
        share_mem_type = SharedMemType::VA_SURFACE;
    } else if (str == "DX_BUFFER") {
        share_mem_type = SharedMemType::DX_BUFFER;
    } else {
        OPENVINO_THROW("Unsupported memory type: ", str);
    }
    return is;
}
/** @endcond */

/**
 * @brief This key identifies type of internal shared memory
 * in a shared memory blob parameter map.
 * @ingroup ov_runtime_ocl_gpu_cpp_api
 */
static constexpr Property<SharedMemType> shared_mem_type{"SHARED_MEM_TYPE"};

/**
 * @brief This key identifies OpenCL memory handle
 * in a shared memory blob parameter map
 * @ingroup ov_runtime_ocl_gpu_cpp_api
 */
static constexpr Property<gpu_handle_param> mem_handle{"MEM_HANDLE"};

/**
 * @brief This key identifies video decoder surface handle
 * in a shared memory blob parameter map
 * @ingroup ov_runtime_ocl_gpu_cpp_api
 */
#ifdef _WIN32
static constexpr Property<gpu_handle_param> dev_object_handle{"DEV_OBJECT_HANDLE"};
#else
static constexpr Property<uint32_t> dev_object_handle{"DEV_OBJECT_HANDLE"};
#endif

/**
 * @brief This key identifies video decoder surface plane
 * in a shared memory blob parameter map
 * @ingroup ov_runtime_ocl_gpu_cpp_api
 */
static constexpr Property<uint32_t> va_plane{"VA_PLANE"};

}  // namespace intel_gpu
}  // namespace ov
