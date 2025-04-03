/*******************************************************************************
* Copyright 2022-2024 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

/// @file
/// C API common types definitions

#ifndef ONEAPI_DNNL_DNNL_COMMON_TYPES_H
#define ONEAPI_DNNL_DNNL_COMMON_TYPES_H

#ifdef __cplusplus
extern "C" {
#endif

/// @cond DO_NOT_DOCUMENT_THIS
#include <stddef.h>
#include <stdint.h>

#include "oneapi/dnnl/dnnl_config.h"

/// @endcond

/// @addtogroup dnnl_api oneDNN API
/// @{

/// @addtogroup dnnl_api_common Common API
/// @{

/// @addtogroup dnnl_api_utils
/// @{

/// Status values returned by the library functions.
typedef enum {
    /// The operation was successful
    dnnl_success = 0,
    /// The operation failed due to an out-of-memory condition
    dnnl_out_of_memory = 1,
    /// The operation failed because of incorrect function arguments
    dnnl_invalid_arguments = 2,
    /// The operation failed because requested functionality is not implemented
    dnnl_unimplemented = 3,
    /// The last available implementation is reached
    dnnl_last_impl_reached = 4,
    /// Primitive or engine failed on execution
    dnnl_runtime_error = 5,
    /// Queried element is not required for given primitive
    dnnl_not_required = 6,
    /// The graph is not legitimate
    dnnl_invalid_graph = 7,
    /// The operation is not legitimate according to op schema
    dnnl_invalid_graph_op = 8,
    /// The shape cannot be inferred or compiled
    dnnl_invalid_shape = 9,
    /// The data type cannot be inferred or compiled
    dnnl_invalid_data_type = 10,
} dnnl_status_t;

/// @} dnnl_api_utils

/// @addtogroup dnnl_api_data_types Data types
/// @{

/// Data type specification
typedef enum {
    /// Undefined data type, used for empty memory descriptors.
    dnnl_data_type_undef = 0,
    /// 16-bit/half-precision floating point.
    dnnl_f16 = 1,
    /// non-standard 16-bit (bfloat16 w/ 7 bit mantissa) floating point.
    dnnl_bf16 = 2,
    /// 32-bit/single-precision floating point.
    dnnl_f32 = 3,
    /// 32-bit signed integer.
    dnnl_s32 = 4,
    /// 8-bit signed integer.
    dnnl_s8 = 5,
    /// 8-bit unsigned integer.
    dnnl_u8 = 6,
    /// 64-bit/double-precision floating point.
    dnnl_f64 = 7,
    /// Boolean data type. Size is C++ implementation defined.
    dnnl_boolean = 8,
    /// [OFP8 standard 8-bit floating-point](https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-0-2023-06-20-pdf)
    /// with a 5-bit exponent and a 2-bit mantissa.
    dnnl_f8_e5m2 = 9,
    /// [OFP8 standard 8-bit floating-point](https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-0-2023-06-20-pdf)
    /// with a 4-bit exponent and a 3-bit mantissa.
    dnnl_f8_e4m3 = 10,
    /// 4-bit signed integer.
    dnnl_s4 = 11,
    /// 4-bit unsigned integer.
    dnnl_u4 = 12,

    /// Parameter to allow internal only data_types without undefined behavior.
    /// This parameter is chosen to be valid for so long as sizeof(int) >= 2.
    dnnl_data_type_max = 0x7fff,
} dnnl_data_type_t;

/// Maximum number of dimensions a tensor can have. Only restricts the amount
/// of space used for the tensor description. Individual computational
/// primitives may support only tensors of certain dimensions.
#define DNNL_MAX_NDIMS 12

/// A type to describe tensor dimension.
typedef int64_t dnnl_dim_t;

/// A type to describe tensor dimensions.
typedef dnnl_dim_t dnnl_dims_t[DNNL_MAX_NDIMS];

/// @} dnnl_api_data_types

/// @addtogroup dnnl_api_fpmath_mode Floating-point Math Mode
/// @{

/// Floating-point math mode
typedef enum {
    /// Default behavior, no downconversions allowed
    dnnl_fpmath_mode_strict,
    /// Implicit f32->bf16 conversions allowed
    dnnl_fpmath_mode_bf16,
    /// Implicit f32->f16 conversions allowed
    dnnl_fpmath_mode_f16,
    /// Implicit f32->f16, f32->tf32 or f32->bf16 conversions allowed
    dnnl_fpmath_mode_any,
    /// Implicit f32->tf32 conversions allowed
    dnnl_fpmath_mode_tf32,
} dnnl_fpmath_mode_t;

/// @} dnnl_api_fpmath_mode

/// @addtogroup dnnl_api_accumulation_mode Accumulation Mode
/// @{

/// Accumulation mode
typedef enum {
    /// Default behavior, f32/f64 for floating point computation, s32
    /// for integer
    dnnl_accumulation_mode_strict,
    /// Same as strict but allows some partial accumulators to be
    /// rounded to src/dst datatype in memory.
    dnnl_accumulation_mode_relaxed,
    /// uses fastest implementation, could use src/dst datatype or
    /// wider datatype for accumulators
    dnnl_accumulation_mode_any,
    /// use s32 accumulators during computation
    dnnl_accumulation_mode_s32,
    /// use f32 accumulators during computation
    dnnl_accumulation_mode_f32,
    /// use f16 accumulators during computation
    dnnl_accumulation_mode_f16
} dnnl_accumulation_mode_t;

/// @} dnnl_api_accumulation_mode

/// @addtogroup dnnl_api_engine Engine
/// @{

/// @brief Kinds of engines.
typedef enum {
    /// An unspecified engine.
    dnnl_any_engine,
    /// CPU engine.
    dnnl_cpu,
    /// GPU engine.
    dnnl_gpu,
} dnnl_engine_kind_t;

/// @struct dnnl_engine
/// @brief An opaque structure to describe an engine.
struct dnnl_engine;
/// @brief An engine handle.
typedef struct dnnl_engine *dnnl_engine_t;
#if 0
// FIXME: looks like this never happens
/// @brief A constant engine handle.
typedef const struct dnnl_engine *const_dnnl_engine_t;
#endif

/// @} dnnl_api_engine

/// @addtogroup dnnl_api_stream Stream
/// @{

/// @brief Stream flags.
typedef enum {
    // In-order execution.
    dnnl_stream_in_order = 0x1U,
    /// Out-of-order execution.
    dnnl_stream_out_of_order = 0x2U,
    /// Default stream configuration.
    dnnl_stream_default_flags = dnnl_stream_in_order,
#ifdef DNNL_EXPERIMENTAL_PROFILING
    /// Enables profiling capabilities.
    dnnl_stream_profiling = 0x4U,
#endif
} dnnl_stream_flags_t;

/// @struct dnnl_stream
/// An opaque structure to describe an execution stream.
struct dnnl_stream;
/// An execution stream handle.
typedef struct dnnl_stream *dnnl_stream_t;
/// A constant execution stream handle.
typedef const struct dnnl_stream *const_dnnl_stream_t;

/// @} dnnl_api_stream

/// @addtogroup dnnl_api_service
/// @{

/// Structure containing version information as per [Semantic
/// Versioning](https://semver.org)
typedef struct {
    int major; ///< Major version
    int minor; ///< Minor version
    int patch; ///< Patch version
    const char *hash; ///< Git hash of the sources (may be absent)
    unsigned cpu_runtime; ///< CPU runtime
    unsigned gpu_runtime; ///< GPU runtime
} dnnl_version_t;

/// @} dnnl_api_service

/// @addtogroup dnnl_api_memory
/// @{

/// Special pointer value that indicates that a memory object should not have
/// an underlying buffer.
#define DNNL_MEMORY_NONE (NULL)

/// Special pointer value that indicates that the library needs to allocate an
/// underlying buffer for a memory object.
#define DNNL_MEMORY_ALLOCATE ((void *)(size_t)-1)

/// @} dnnl_api_memory

/// @} dnnl_api_common

/// @} dnnl_api

#ifdef __cplusplus
}
#endif

#endif
