// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header file for the OpenVINO Runtime tensor API.
 *
 * @file openvino/runtime/remote_tensor.hpp
 */
#pragma once

#include "openvino/runtime/common.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov {

class RemoteContext;

/**
 * @brief Remote memory access and interoperability API.
 * @ingroup ov_runtime_cpp_api
 */
class OPENVINO_RUNTIME_API RemoteTensor : public Tensor {
    using Tensor::Tensor;

public:
    /// @brief Default constructor
    RemoteTensor() = default;

    /**
     * @brief Constructs region of interest (ROI) tensor from another remote tensor.
     * @note Does not perform memory allocation internally
     * @param other original tensor
     * @param begin start coordinate of ROI object inside of the original object.
     * @param end end coordinate of ROI object inside of the original object.
     * @note A Number of dimensions in `begin` and `end` must match number of dimensions in `other.get_shape()`
     */
    RemoteTensor(const RemoteTensor& other, const Coordinate& begin, const Coordinate& end);

    /**
     * @brief Checks OpenVINO remote type.
     * @param tensor Tensor which type is checked.
     * @param type_info Map with remote object runtime info.
     * @throw Exception if type check with specified parameters failed.
     */
    static void type_check(const Tensor& tensor, const std::map<std::string, std::vector<std::string>>& type_info = {});

    /**
     * @brief Access to host memory is not available for RemoteTensor.
     * To access a device-specific memory, cast to a specific RemoteTensor derived object and work with its
     * properties or parse device memory properties via RemoteTensor::get_params.
     * @return Nothing, throws an exception.
     */
    void* data(const element::Type) = delete;

    template <typename T>
    T* data() = delete;

    /**
     * @brief Copies data from this RemoteTensor to the specified destination tensor.
     * @param dst The destination tensor to which data will be copied.
     */
    void copy_to(ov::Tensor& dst) const;

    /**
     * @brief Copies data from the specified source tensor to this RemoteTensor.
     * @param src The source tensor from which data will be copied.
     */
    void copy_from(const ov::Tensor& src);

    /**
     * @brief Returns a map of device-specific parameters required for low-level
     * operations with underlying object.
     * Parameters include device/context/surface/buffer handles, access flags,
     * etc. Content of the returned map depends on remote execution context that is
     * currently set on the device (working scenario).
     * Abstract method.
     * @return A map of name/parameter elements.
     */
    ov::AnyMap get_params() const;

    /**
     * @brief Returns name of a device on which the underlying object is allocated.
     * Abstract method.
     * @return A device name string in fully specified format `<device_name>[.<device_id>[.<tile_id>]]`.
     */
    std::string get_device_name() const;
};

}  // namespace ov
