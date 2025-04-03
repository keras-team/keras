// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header for advanced hardware related properties for NPU plugin
 *        To use in set_property, compile_model, import_model, get_property methods
 *
 * @file openvino/runtime/intel_npu/properties.hpp
 */
#pragma once

#include "openvino/runtime/properties.hpp"

namespace ov {

/**
 * @defgroup ov_runtime_npu_prop_cpp_api Intel NPU specific properties
 * @ingroup ov_runtime_cpp_api
 * Set of Intel NPU specific properties.
 */

/**
 * @brief Namespace with Intel NPU specific properties
 */
namespace intel_npu {

/**
 * @brief [Only for NPU plugin]
 * Type: uint64_t
 * Read-only property to get size of already allocated NPU DDR memory (both for discrete/integrated NPU devices)
 *
 * Note: Queries driver both for discrete/integrated NPU devices
 * @ingroup ov_runtime_npu_prop_cpp_api
 */
static constexpr ov::Property<uint64_t, ov::PropertyMutability::RO> device_alloc_mem_size{"NPU_DEVICE_ALLOC_MEM_SIZE"};

/**
 * @brief [Only for NPU plugin]
 * Type: uint64_t
 * Read-only property to get size of available NPU DDR memory (both for discrete/integrated NPU devices)
 *
 * Note: Queries driver both for discrete/integrated NPU devices
 * @ingroup ov_runtime_npu_prop_cpp_api
 */
static constexpr ov::Property<uint64_t, ov::PropertyMutability::RO> device_total_mem_size{"NPU_DEVICE_TOTAL_MEM_SIZE"};

/**
 * @brief [Only for NPU plugin]
 * Type: uint32_t
 * Read-only property to get NPU driver version (for both discrete/integrated NPU devices)
 * @ingroup ov_runtime_npu_prop_cpp_api
 */
static constexpr ov::Property<uint32_t, ov::PropertyMutability::RO> driver_version{"NPU_DRIVER_VERSION"};

/**
 * @brief [Only for NPU plugin]
 * Type: uint32_t
 * Read-only property to get NPU compiler version. Composite of Major (16bit MSB) and Minor (16bit LSB)
 * @ingroup ov_runtime_npu_prop_cpp_api
 */
static constexpr ov::Property<uint32_t, ov::PropertyMutability::RO> compiler_version{"NPU_COMPILER_VERSION"};

/**
 * @brief [Only for NPU compiler]
 * Type: std::string
 * Set various parameters supported by the NPU compiler.
 * @ingroup ov_runtime_npu_prop_cpp_api
 */
static constexpr ov::Property<std::string> compilation_mode_params{"NPU_COMPILATION_MODE_PARAMS"};

/**
 * @brief [Only for NPU compiler]
 * Type: boolean
 * Set or verify state of dynamic quantization in  the NPU compiler
 * @ingroup ov_runtime_npu_prop_cpp_api
 */
static constexpr ov::Property<bool> compiler_dynamic_quantization{"NPU_COMPILER_DYNAMIC_QUANTIZATION"};

/**
 * @brief [Only for NPU plugin]
 * Type: std::bool
 * Set turbo on or off. The turbo mode, where available, provides a hint to the system to maintain the
 * maximum NPU frequency and memory throughput within the platform TDP limits.
 * Turbo mode is not recommended for sustainable workloads due to higher power consumption and potential impact on other
 * compute resources.
 * @ingroup ov_runtime_npu_prop_cpp_api
 */
static constexpr ov::Property<bool> turbo{"NPU_TURBO"};

/**
 * @brief [Only for NPU Compiler]
 * Type: integer, default is -1
 * Sets the number of npu tiles to compile the model for.
 */
static constexpr ov::Property<int64_t> tiles{"NPU_TILES"};

/**
 * @brief
 * Type: integer, default is -1
 * Maximum number of tiles supported by the device we compile for. Can be set for offline compilation. If not set, it
 * will be populated by driver.
 */
static constexpr ov::Property<int64_t> max_tiles{"NPU_MAX_TILES"};

/**
 * @brief [Only for NPU plugin]
 * Type: std::bool
 * Bypass caching of the compiled model by UMD cache.
 * @ingroup ov_runtime_npu_prop_cpp_api
 */
static constexpr ov::Property<bool> bypass_umd_caching{"NPU_BYPASS_UMD_CACHING"};

/**
 * @brief [Only for NPU Plugin]
 * Type: boolean, default is false
 * This option allows to delay loading the weights until inference is created
 */
static constexpr ov::Property<bool> defer_weights_load{"NPU_DEFER_WEIGHTS_LOAD"};

}  // namespace intel_npu
}  // namespace ov
