// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief a header that defines wrappers for internal GPU plugin-specific
 * shared Video Acceleration device contexts
 * and shared memory tensors which contain Video Acceleration surfaces
 *
 * @file openvino/runtime/intel_gpu/ocl/va.hpp
 */
#pragma once

#ifdef _WIN32
#    if !defined(__MINGW32__) && !defined(__MINGW64__)
#        error "OpenCL VA-API interoperability is supported only on Linux-based platforms"
#    endif
#endif

#include <memory>
#include <string>

#include "openvino/runtime/intel_gpu/ocl/ocl.hpp"

// clang-format off
#include <va/va.h>
// clang-format on

namespace ov {
namespace intel_gpu {
namespace ocl {

/**
 * @brief This class represents an abstraction for GPU plugin remote tensor
 * which is shared with VA output surface.
 * The plugin object derived from this class can be obtained with VAContext::create_tensor() call.
 * @note User can also obtain OpenCL 2D image handle from this class.
 * @ingroup ov_runtime_ocl_gpu_cpp_api
 */
class VASurfaceTensor : public ClImage2DTensor {
public:
    /**
     * @brief Checks that type defined runtime parameters are presented in remote object
     * @param tensor a tensor to check
     */
    static void type_check(const Tensor& tensor) {
        RemoteTensor::type_check(tensor,
                                 {{std::string(ov::intel_gpu::dev_object_handle.name()), {}},
                                  {std::string(ov::intel_gpu::va_plane.name()), {}},
                                  {std::string(ov::intel_gpu::shared_mem_type.name()),
                                   {ov::Any(ov::intel_gpu::SharedMemType::VA_SURFACE).as<std::string>()}}});
    }
    /**
     * @brief VASurfaceID conversion operator for the VASurfaceTensor object.
     * @return `VASurfaceID` handle
     */
    operator VASurfaceID() {
        return static_cast<VASurfaceID>(get_params().at(ov::intel_gpu::dev_object_handle.name()).as<uint32_t>());
    }

    /**
     * @brief Returns plane ID of underlying video decoder surface
     * @return Plane ID
     */
    uint32_t plane() {
        return get_params().at(ov::intel_gpu::va_plane.name()).as<uint32_t>();
    }
};

/**
 * @brief This class represents an abstraction for GPU plugin remote context
 * which is shared with VA display object.
 * The plugin object derived from this class can be obtained either with
 * CompiledModel::get_context() or Core::create_context() calls.
 * @note User can also obtain OpenCL context handle from this class.
 * @ingroup ov_runtime_ocl_gpu_cpp_api
 */
class VAContext : public ClContext {
public:
    // Needed to make create_tensor overloads from base class visible for user
    using ClContext::create_tensor;

    /**
     * @brief Checks that type defined runtime parameters are presented in remote object
     * @param remote_context A remote context to check
     */
    static void type_check(const RemoteContext& remote_context) {
        RemoteContext::type_check(remote_context,
                                  {{std::string(ov::intel_gpu::va_device.name()), {}},
                                   {std::string(ov::intel_gpu::context_type.name()),
                                    {ov::Any(ov::intel_gpu::ContextType::VA_SHARED).as<std::string>()}}});
    }

    /**
     * @brief `VADisplay` conversion operator for the VAContext object.
     * @return Underlying `VADisplay` object handle
     */
    operator VADisplay() {
        return static_cast<VADisplay>(get_params().at(ov::intel_gpu::va_device.name()).as<gpu_handle_param>());
    }

    /**
     * @brief Constructs remote context object from VA display handle
     * @param core OpenVINO Runtime Core object
     * @param device A `VADisplay` to create remote context from
     * @param target_tile_id Desired tile id within given context for multi-tile system. Default value (-1) means
     * that root device should be used
     */
    VAContext(Core& core, VADisplay device, int target_tile_id = -1) : ClContext() {
        AnyMap context_params = {{ov::intel_gpu::context_type.name(), ov::intel_gpu::ContextType::VA_SHARED},
                                 {ov::intel_gpu::va_device.name(), static_cast<gpu_handle_param>(device)},
                                 {ov::intel_gpu::tile_id.name(), target_tile_id}};
        *this = core.create_context(device_name, context_params).as<VAContext>();
    }

    /**
     * @brief This function is used to obtain a NV12 tensor from NV12 VA decoder output.
     * The resulting tensor contains two remote tensors for Y and UV planes of the surface.
     * @param height A height of Y plane
     * @param width A width of Y plane
     * @param nv12_surf NV12 `VASurfaceID` to create NV12 from
     * @return A pair of remote tensors for each plane
     */
    std::pair<VASurfaceTensor, VASurfaceTensor> create_tensor_nv12(const size_t height,
                                                                   const size_t width,
                                                                   const VASurfaceID nv12_surf) {
        AnyMap tensor_params = {{ov::intel_gpu::shared_mem_type.name(), ov::intel_gpu::SharedMemType::VA_SURFACE},
                                {ov::intel_gpu::dev_object_handle.name(), nv12_surf},
                                {ov::intel_gpu::va_plane.name(), uint32_t(0)}};
        auto y_tensor = create_tensor(element::u8, {1, height, width, 1}, tensor_params);
        tensor_params[ov::intel_gpu::va_plane.name()] = uint32_t(1);
        auto uv_tensor = create_tensor(element::u8, {1, height / 2, width / 2, 2}, tensor_params);
        return std::make_pair(y_tensor.as<VASurfaceTensor>(), uv_tensor.as<VASurfaceTensor>());
    }

    /**
     * @brief This function is used to create remote tensor from VA surface handle
     * @param type Tensor element type
     * @param shape Tensor shape
     * @param surface A `VASurfaceID` to create remote tensor from
     * @param plane An index of a plane inside `VASurfaceID` to create tensor from
     * @return A remote tensor wrapping `VASurfaceID`
     */
    inline VASurfaceTensor create_tensor(const element::Type type,
                                         const Shape& shape,
                                         const VASurfaceID surface,
                                         const uint32_t plane = 0) {
        AnyMap params = {{ov::intel_gpu::shared_mem_type.name(), ov::intel_gpu::SharedMemType::VA_SURFACE},
                         {ov::intel_gpu::dev_object_handle.name(), surface},
                         {ov::intel_gpu::va_plane.name(), plane}};
        return create_tensor(type, shape, params).as<VASurfaceTensor>();
    }
};
}  // namespace ocl
}  // namespace intel_gpu
}  // namespace ov
