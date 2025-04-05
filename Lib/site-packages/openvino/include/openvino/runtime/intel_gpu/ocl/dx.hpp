// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header that defines wrappers for internal GPU plugin-specific
 * shared Video Acceleration device contexts
 * and shared memory tensors which contain Video Acceleration surfaces
 *
 * @file openvino/runtime/intel_gpu/ocl/dx.hpp
 */
#pragma once

#ifndef NOMINMAX
#    define NOMINMAX
#endif

#ifndef _WIN32
#    error "OpenCL DirectX interoperability is supported only on Windows platforms"
#endif

#include <d3d11.h>

#include <memory>
#include <string>

#include "openvino/runtime/intel_gpu/ocl/ocl.hpp"
#include "openvino/runtime/intel_gpu/remote_properties.hpp"

namespace ov {
namespace intel_gpu {
namespace ocl {

/**
 * @brief This class represents an abstraction for GPU plugin remote tensor
 * which is shared with Direct3D 11 buffer.
 * The plugin object derived from this class can be obtained with D3DContext::create_tensor() call.
 * @note User can also obtain OpenCL buffer handle from this class.
 * @ingroup ov_runtime_ocl_gpu_cpp_api
 */
class D3DBufferTensor : public ClBufferTensor {
public:
    /**
     * @brief Checks that type defined runtime parameters are presented in remote object
     * @param tensor a tensor to check
     */
    static void type_check(const Tensor& tensor) {
        RemoteTensor::type_check(tensor,
                                 {{std::string(ov::intel_gpu::dev_object_handle.name()), {}},
                                  {std::string(ov::intel_gpu::shared_mem_type.name()),
                                   {ov::Any(ov::intel_gpu::SharedMemType::DX_BUFFER).as<std::string>()}}});
    }

    /**
     * @brief ID3D11Buffer conversion operator for the D3DContext object.
     * @return Pointer to underlying ID3D11Buffer interface
     */
    operator ID3D11Buffer*() {
        return static_cast<ID3D11Buffer*>(
            get_params().at(ov::intel_gpu::dev_object_handle.name()).as<gpu_handle_param>());
    }
};

/**
 * @brief This class represents an abstraction for GPU plugin remote tensor
 * which is shared with Direct3D 11 2D texture.
 * The plugin object derived from this class can be obtained with D3DContext::create_tensor() call.
 * @note User can also obtain OpenCL 2D image handle from this class.
 * @ingroup ov_runtime_ocl_gpu_cpp_api
 */
class D3DSurface2DTensor : public ClImage2DTensor {
public:
    /**
     * @brief Checks that type defined runtime parameters are presented in remote object
     * @param remote_tensor remote tensor to check
     */
    static void type_check(const Tensor& remote_tensor) {
        RemoteTensor::type_check(remote_tensor,
                                 {{std::string(ov::intel_gpu::dev_object_handle.name()), {}},
                                  {std::string(ov::intel_gpu::va_plane.name()), {}},
                                  {std::string(ov::intel_gpu::shared_mem_type.name()),
                                   {ov::Any(ov::intel_gpu::SharedMemType::VA_SURFACE).as<std::string>()}}});
    }

    /**
     * @brief ID3D11Texture2D conversion operator for the D3DContext object.
     * @return Pointer to underlying ID3D11Texture2D interface
     */
    operator ID3D11Texture2D*() {
        return static_cast<ID3D11Texture2D*>(
            get_params().at(ov::intel_gpu::dev_object_handle.name()).as<gpu_handle_param>());
    }

    /**
     * @brief Returns plane ID of underlying video decoder surface, or 0 if no video surface was shared.
     * @return Plane ID
     */
    uint32_t plane() {
        return get_params().at(ov::intel_gpu::va_plane.name()).as<uint32_t>();
    }
};

/**
 * @brief This class represents an abstraction for GPU plugin remote context
 * which is shared with Direct3D 11 device.
 * The plugin object derived from this class can be obtained either with
 * CompiledModel::get_context() or Core::create_context() calls.
 * @note User can also obtain OpenCL context handle from this class.
 * @ingroup ov_runtime_ocl_gpu_cpp_api
 */
class D3DContext : public ClContext {
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
     * @brief ID3D11Device conversion operator for the D3DContext object.
     * @return Pointer to underlying ID3D11Device interface
     */
    operator ID3D11Device*() {
        return static_cast<ID3D11Device*>(get_params().at(ov::intel_gpu::va_device.name()).as<gpu_handle_param>());
    }

    /**
     * @brief Constructs D3DContext remote context object from ID3D11Device
     * @param core OpenVINO Runtime Core object instance
     * @param device A pointer to ID3D11Device to be used to create a remote context
     * @param target_tile_id Desired tile id within given context for multi-tile system. Default value (-1) means
     * that root device should be used
     */
    D3DContext(Core& core, ID3D11Device* device, int target_tile_id = -1) : ClContext() {
        // clang-format off
        AnyMap context_params = {
            {ov::intel_gpu::context_type.name(), ov::intel_gpu::ContextType::VA_SHARED},
            {ov::intel_gpu::va_device.name(), static_cast<gpu_handle_param>(device)},
            {ov::intel_gpu::tile_id.name(), target_tile_id}
        };
        *this = core.create_context(device_name, context_params).as<D3DContext>();
    }

    /**
     * @brief This function is used to obtain a NV12 tensor from NV12 DXGI video decoder output.
     * The resulting tensor contains two remote tensors for Y and UV planes of the surface.
     * @param height Height of Y plane
     * @param width Width of Y plane
     * @param nv12_surf A ID3D11Texture2D instance to create NV12 tensor from
     * @return A pair of remote tensors for each plane
     */
    std::pair<D3DSurface2DTensor, D3DSurface2DTensor> create_tensor_nv12(const size_t height, const size_t width, ID3D11Texture2D* nv12_surf) {
        AnyMap tensor_params = {{ov::intel_gpu::shared_mem_type.name(), ov::intel_gpu::SharedMemType::VA_SURFACE},
                                  {ov::intel_gpu::dev_object_handle.name(), static_cast<gpu_handle_param>(nv12_surf)},
                                  {ov::intel_gpu::va_plane.name(), uint32_t(0)}};
        auto y_tensor = create_tensor(element::u8, {1, height, width, 1}, tensor_params);
        tensor_params[ov::intel_gpu::mem_handle.name()] = static_cast<gpu_handle_param>(nv12_surf);
        tensor_params[ov::intel_gpu::va_plane.name()] = uint32_t(1);
        auto uv_tensor = create_tensor(element::u8, {1, height / 2, width / 2, 2}, tensor_params);
        return std::make_pair(y_tensor.as<D3DSurface2DTensor>(), uv_tensor.as<D3DSurface2DTensor>());
    }

    /**
     * @brief This function is used to obtain remote tensor object from ID3D11Buffer
     * @param type Tensor element type
     * @param shape Tensor shape
     * @param buffer A pointer to ID3D11Buffer instance to create remote tensor based on
     * @return A remote tensor instance
     */
    D3DBufferTensor create_tensor(const element::Type type, const Shape& shape, ID3D11Buffer* buffer) {
        AnyMap params = {{ov::intel_gpu::shared_mem_type.name(), ov::intel_gpu::SharedMemType::DX_BUFFER},
                           {ov::intel_gpu::dev_object_handle.name(), static_cast<gpu_handle_param>(buffer)}};
        return create_tensor(type, shape, params).as<D3DBufferTensor>();
    }

    /**
     * @brief This function is used to obtain remote tensor object from ID3D11Texture2D
     * @param type Tensor element type
     * @param shape Tensor shape
     * @param surface Pointer to ID3D11Texture2D interface of the objects that owns NV12 texture
     * @param plane ID of the plane to be shared (0 or 1)
     * @return D3DSurface2DTensor tensor
     * @note The underlying ID3D11Texture2D can also be a plane of output surface of DXGI video decoder
     */
    D3DSurface2DTensor create_tensor(const element::Type type,
                                     const Shape& shape,
                                     ID3D11Texture2D* surface,
                                     uint32_t plane = 0) {
        AnyMap params = {{ov::intel_gpu::shared_mem_type.name(), ov::intel_gpu::SharedMemType::VA_SURFACE},
                           {ov::intel_gpu::dev_object_handle.name(), static_cast<gpu_handle_param>(surface)},
                           {ov::intel_gpu::va_plane.name(), plane}};
        return create_tensor(type, shape, params).as<D3DSurface2DTensor>();
    }
};
}  // namespace ocl
}  // namespace intel_gpu
}  // namespace ov
