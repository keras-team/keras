// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header for properties of shared device contexts and shared device memory tensors for NPU device
 *        To use in constructors of Remote objects
 *
 * @file openvino/runtime/intel_npu/remote_properties.hpp
 */
#pragma once

#include "openvino/runtime/properties.hpp"

namespace ov {
namespace intel_npu {

using npu_handle_param = void*;

/**
 * @brief Enum to define the type of the shared memory buffer
 * @ingroup ov_runtime_level_zero_npu_cpp_api
 */
enum class MemType {
    L0_INTERNAL_BUF = 0,  //!< Internal Level Zero buffer type allocated by plugin
    SHARED_BUF = 1,       //!< Shared buffer
};

/** @cond INTERNAL */
inline std::ostream& operator<<(std::ostream& os, const MemType& mem_type) {
    switch (mem_type) {
    case MemType::L0_INTERNAL_BUF:
        return os << "L0_INTERNAL_BUF";
    case MemType::SHARED_BUF:
        return os << "SHARED_BUF";
    default:
        OPENVINO_THROW("Unsupported memory type");
    }
}

inline std::istream& operator>>(std::istream& is, MemType& mem_type) {
    std::string str;
    is >> str;
    if (str == "L0_INTERNAL_BUF") {
        mem_type = MemType::L0_INTERNAL_BUF;
    } else if (str == "SHARED_BUF") {
        mem_type = MemType::SHARED_BUF;
    } else {
        OPENVINO_THROW("Unsupported memory type: ", str);
    }
    return is;
}
/** @endcond */

/**
 * @brief This key identifies type of internal shared memory
 * in a shared memory tensor parameter map.
 * @ingroup ov_runtime_level_zero_npu_cpp_api
 */
static constexpr Property<MemType> mem_type{"MEM_TYPE"};

/**
 * @brief This key identifies memory handle
 * in a shared memory tensor parameter map
 * @ingroup ov_runtime_level_zero_npu_cpp_api
 */
static constexpr Property<npu_handle_param> mem_handle{"MEM_HANDLE"};

/**
 * @brief This key identifies LevelZero context handle
 * in a shared context parameter map
 * @ingroup ov_runtime_level_zero_npu_cpp_api
 */
static constexpr Property<npu_handle_param> l0_context{"L0_CONTEXT"};

/**
 * @brief Enum to define the type of the tensor
 * @ingroup ov_runtime_level_zero_npu_cpp_api
 */
enum class TensorType {
    INPUT = 0,   //!< Tensor is only used as input
    OUTPUT = 1,  //!< Tensor is only used as output
    BINDED = 2   //!< Tensor could be used as input and output
};

/** @cond INTERNAL */
inline std::ostream& operator<<(std::ostream& os, const TensorType& tensor_type) {
    switch (tensor_type) {
    case TensorType::INPUT:
        return os << "INPUT";
    case TensorType::OUTPUT:
        return os << "OUTPUT";
    case TensorType::BINDED:
        return os << "BINDED";
    default:
        OPENVINO_THROW("Unsupported tensor type");
    }
}

inline std::istream& operator>>(std::istream& is, TensorType& tensor_type) {
    std::string str;
    is >> str;
    if (str == "INPUT") {
        tensor_type = TensorType::INPUT;
    } else if (str == "OUTPUT") {
        tensor_type = TensorType::OUTPUT;
    } else if (str == "BINDED") {
        tensor_type = TensorType::BINDED;
    } else {
        OPENVINO_THROW("Unsupported tensor type: ", str);
    }
    return is;
}
/** @endcond */

/**
 * @brief This key sets the type of the internal Level Zero buffer
 * allocated by the plugin in a shared memory tensor parameter map.
 * @ingroup ov_runtime_level_zero_npu_cpp_api
 */
static constexpr Property<TensorType> tensor_type{"TENSOR_TYPE"};

}  // namespace intel_npu
}  // namespace ov
