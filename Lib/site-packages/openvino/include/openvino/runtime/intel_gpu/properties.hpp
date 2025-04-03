// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header for advanced hardware related properties for GPU plugin
 *        To use in set_property, compile_model, import_model, get_property methods
 *
 * @file openvino/runtime/intel_gpu/properties.hpp
 */
#pragma once

#include "openvino/runtime/properties.hpp"

namespace ov {

/**
 * @defgroup ov_runtime_ocl_gpu_prop_cpp_api Intel GPU OpenCL specific properties
 * @ingroup ov_runtime_cpp_api
 * Set of Intel GPU OpenCL specific properties.
 */

/**
 * @brief Namespace with Intel GPU specific properties
 */
namespace intel_gpu {

/**
 * @brief Read-only property which defines size of memory in bytes available for the device. For iGPU it returns host
 * memory size, for dGPU - dedicated gpu memory size
 * @ingroup ov_runtime_ocl_gpu_prop_cpp_api
 */
static constexpr Property<uint64_t, PropertyMutability::RO> device_total_mem_size{"GPU_DEVICE_TOTAL_MEM_SIZE"};

/**
 * @brief Read-only property to get microarchitecture identifier in major.minor.revision format
 * @ingroup ov_runtime_ocl_gpu_prop_cpp_api
 */
static constexpr Property<std::string, PropertyMutability::RO> uarch_version{"GPU_UARCH_VERSION"};

/**
 * @brief Read-only property to get count of execution units for current GPU
 * @ingroup ov_runtime_ocl_gpu_prop_cpp_api
 */
static constexpr Property<int32_t, PropertyMutability::RO> execution_units_count{"GPU_EXECUTION_UNITS_COUNT"};

/**
 * @brief Read-only property to get statistics of GPU memory allocated by engine for each allocation type
 * It contains information about current memory usage
 * @ingroup ov_runtime_ocl_gpu_prop_cpp_api
 */
static constexpr Property<std::map<std::string, uint64_t>, PropertyMutability::RO> memory_statistics{
    "GPU_MEMORY_STATISTICS"};

/**
 * @brief Turning on this key enables to unroll recurrent layers such as TensorIterator or Loop with fixed iteration
 * count. This key is turned on by default. Turning this key on will achieve better inference performance for loops with
 * not too many iteration counts (less than 16, as a rule of thumb). Turning this key off will achieve better
 * performance for both graph loading time and inference time with many iteration counts (greater than 16). Note that
 * turning this key on will increase the graph loading time in proportion to the iteration counts.
 * Thus, this key should be turned off if graph loading time is considered to be most important target to optimize.
 * @ingroup ov_runtime_ocl_gpu_prop_cpp_api
 */
static constexpr Property<bool> enable_loop_unrolling{"GPU_ENABLE_LOOP_UNROLLING"};

/**
 * @brief Turning on this key disables winograd convolution.
 * Winograd convolution has different characteristics for accuracy and performance compared to other convolution
 * implementations.
 * @ingroup ov_runtime_ocl_gpu_prop_cpp_api
 */
static constexpr Property<bool> disable_winograd_convolution{"GPU_DISABLE_WINOGRAD_CONVOLUTION"};

namespace hint {
/**
 * @brief This enum represents the possible value of ov::intel_gpu::hint::queue_throttle property:
 * - LOW is used for CL_QUEUE_THROTTLE_LOW_KHR OpenCL throttle hint
 * - MEDIUM (DEFAULT) is used for CL_QUEUE_THROTTLE_MED_KHR OpenCL throttle hint
 * - HIGH is used for CL_QUEUE_THROTTLE_HIGH_KHR OpenCL throttle hint
 * @ingroup ov_runtime_ocl_gpu_prop_cpp_api
 */
using ThrottleLevel = ov::hint::Priority;

/**
 * @brief This key instructs the GPU plugin to use OpenCL queue throttle hints
 * as defined in https://www.khronos.org/registry/OpenCL/specs/opencl-2.1-extensions.pdf,
 * chapter 9.19. This option should be used with ov::intel_gpu::hint::ThrottleLevel values.
 * @ingroup ov_runtime_ocl_gpu_prop_cpp_api
 */
static constexpr Property<ThrottleLevel> queue_throttle{"GPU_QUEUE_THROTTLE"};

/**
 * @brief This key instructs the GPU plugin to use the OpenCL queue priority hint
 * as defined in https://www.khronos.org/registry/OpenCL/specs/opencl-2.1-extensions.pdf.
 * This option should be used with ov::hint::Priority:
 * - LOW is used for CL_QUEUE_PRIORITY_LOW_KHR OpenCL priority hint
 * - MEDIUM (DEFAULT) is used for CL_QUEUE_PRIORITY_MED_KHR OpenCL priority hint
 * - HIGH is used for CL_QUEUE_PRIORITY_HIGH_KHR OpenCL priority hint
 * @ingroup ov_runtime_ocl_gpu_prop_cpp_api
 */
static constexpr Property<ov::hint::Priority> queue_priority{"GPU_QUEUE_PRIORITY"};

/**
 * @brief This key instructs the GPU plugin which cpu core type of TBB affinity used in load network.
 * This option has 3 types of levels: HIGH, LOW, and ANY. It is only affected on Hybrid CPUs.
 * - LOW - instructs the GPU Plugin to use LITTLE cores if they are available
 * - MEDIUM (DEFAULT) - instructs the GPU Plugin to use any available cores (BIG or LITTLE cores)
 * - HIGH - instructs the GPU Plugin to use BIG cores if they are available
 * @ingroup ov_runtime_ocl_gpu_prop_cpp_api
 */
static constexpr Property<ov::hint::Priority> host_task_priority{"GPU_HOST_TASK_PRIORITY"};

/**
 * @brief This key identifies available device memory size in bytes
 * @ingroup ov_runtime_ocl_gpu_prop_cpp_api
 */
static constexpr Property<int64_t> available_device_mem{"AVAILABLE_DEVICE_MEM_SIZE"};

/**
 * @brief Turning on this key disables SDPA operation decomposition and keeps SDPA operation in the graph.
 * Enabling SDPA optimization may provide performance improvements and memory usage reduction.
 * This key serves as a recommendation and may be ignored in known sub-optimal cases.
 * @ingroup ov_runtime_ocl_gpu_prop_cpp_api
 */
static constexpr Property<bool> enable_sdpa_optimization{"GPU_ENABLE_SDPA_OPTIMIZATION"};

/**
 * @brief Turning on this property enables kernels reuse between implementations, resulting in a lower memory footprint.
 * However, as a drawback, OpenCL set_arguments() call will be made more often, resulting in higher host pressure
 * and slower execution in some host-bottleneck cases.
 * This property is available only for single-stream scenarios and will be ignored in other cases.
 * @ingroup ov_runtime_ocl_gpu_prop_cpp_api
 */
static constexpr Property<bool> enable_kernels_reuse{"GPU_ENABLE_KERNELS_REUSE"};
}  // namespace hint

/**
 * @brief These keys instruct the GPU plugin to use surface/buffer memory type.
 */
namespace memory_type {

/**
 * @brief Native video decoder surface
 * @ingroup ov_runtime_ocl_gpu_prop_cpp_api
 */
static constexpr auto surface = "GPU_SURFACE";

/*
 * @brief OpenCL buffer
 * @ingroup ov_runtime_ocl_gpu_prop_cpp_api
 */
static constexpr auto buffer = "GPU_BUFFER";

}  // namespace memory_type

/**
 * @brief Possible return value for ov::device::capabilities property
 */
namespace capability {

/**
 * @brief Device has hardware block for matrix multiplication
 * @ingroup ov_runtime_ocl_gpu_prop_cpp_api
 */
constexpr static const auto HW_MATMUL = "GPU_HW_MATMUL";

}  // namespace capability
}  // namespace intel_gpu
}  // namespace ov
