
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header for advanced hardware related properties for CPU device
 *        To use in set_property, compile_model, import_model, get_property methods
 *
 * @file openvino/runtime/intel_cpu/properties.hpp
 */
#pragma once

#include "openvino/runtime/properties.hpp"

namespace ov {

/**
 * @defgroup ov_runtime_cpu_prop_cpp_api Intel CPU specific properties
 * @ingroup ov_runtime_cpp_api
 * Set of Intel CPU specific properties.
 */

/**
 * @brief Namespace with Intel CPU specific properties
 */
namespace intel_cpu {

/**
 * @brief This property define whether to perform denormals optimization.
 * @ingroup ov_runtime_cpu_prop_cpp_api
 *
 * Computation with denormals is very time consuming. FTZ(Flushing denormals to zero) and DAZ(Denormals as zero)
 * could significantly improve the performance, but it does not comply with IEEE standard. In most cases, this behavior
 * has little impact on model accuracy. Users could enable this optimization if no or acceptable accuracy drop is seen.
 * The following code enables denormals optimization
 *
 * @code
 * ie.set_property(ov::denormals_optimization(true)); // enable denormals optimization
 * @endcode
 *
 * The following code disables denormals optimization
 *
 * @code
 * ie.set_property(ov::denormals_optimization(false)); // disable denormals optimization
 * @endcode
 */
static constexpr Property<bool> denormals_optimization{"CPU_DENORMALS_OPTIMIZATION"};

/**
 * @brief This property defines threshold for sparse weights decompression feature activation
 * @ingroup ov_runtime_cpu_prop_cpp_api
 *
 * Sparse weights decompression feature allows to pack weights for Matrix Multiplication operations directly in the CPU
 * plugin at the model compilation stage and store non-zero values in a special packed format. Then, during the
 * execution of the model, the weights are unpacked and used in the computational kernel. Since the weights are loaded
 * from DDR/L3 cache in the packed format this significantly decreases memory consumption and as a consequence improve
 * inference performance. The following code allows to set the sparse rate value.
 *
 * @code
 * core.set_property(ov::intel_cpu::sparse_weights_decompression_rate(0.8));
 * @endcode
 */
static constexpr Property<float> sparse_weights_decompression_rate{"CPU_SPARSE_WEIGHTS_DECOMPRESSION_RATE"};

}  // namespace intel_cpu
}  // namespace ov
