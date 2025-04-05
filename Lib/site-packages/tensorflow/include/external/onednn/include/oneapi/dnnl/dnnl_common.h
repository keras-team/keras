/*******************************************************************************
* Copyright 2022-2023 Intel Corporation
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
/// C common API

#ifndef ONEAPI_DNNL_DNNL_COMMON_H
#define ONEAPI_DNNL_DNNL_COMMON_H

#include "oneapi/dnnl/dnnl_common_types.h"
#include "oneapi/dnnl/dnnl_config.h"
#include "oneapi/dnnl/dnnl_version.h"

#ifdef __cplusplus
extern "C" {
#endif

/// @addtogroup dnnl_api  oneDNN API
/// @{

/// @addtogroup dnnl_api_common Common API
/// @{

/// @addtogroup dnnl_api_engine Engine
/// @{

/// Returns the number of engines of a particular kind.
///
/// @param kind Kind of engines to count.
/// @returns Count of the engines.
size_t DNNL_API dnnl_engine_get_count(dnnl_engine_kind_t kind);

/// Creates an engine.
///
/// @param engine Output engine.
/// @param kind Engine kind.
/// @param index Engine index that should be between 0 and the count of
///     engines of the requested kind.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_engine_create(
        dnnl_engine_t *engine, dnnl_engine_kind_t kind, size_t index);

/// Returns the kind of an engine.
///
/// @param engine Engine to query.
/// @param kind Output engine kind.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_engine_get_kind(
        dnnl_engine_t engine, dnnl_engine_kind_t *kind);

/// Destroys an engine.
///
/// @param engine Engine to destroy.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_engine_destroy(dnnl_engine_t engine);

/// @} dnnl_api_engine

/// @addtogroup dnnl_api_stream Stream
/// @{

/// Creates an execution stream.
///
/// @param stream Output execution stream.
/// @param engine Engine to create the execution stream on.
/// @param flags Stream behavior flags (@sa dnnl_stream_flags_t).
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_stream_create(
        dnnl_stream_t *stream, dnnl_engine_t engine, unsigned flags);

/// Returns the engine of a stream object.
///
/// @param stream Stream object.
/// @param engine Output engine on which the stream is created.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_stream_get_engine(
        const_dnnl_stream_t stream, dnnl_engine_t *engine);

/// Waits for all primitives in the execution stream to finish computations.
///
/// @param stream Execution stream.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_stream_wait(dnnl_stream_t stream);

/// Destroys an execution stream.
///
/// @param stream Execution stream to destroy.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_stream_destroy(dnnl_stream_t stream);

/// @} dnnl_api_stream

/// @addtogroup dnnl_api_fpmath_mode Floating-point Math Mode
/// @{

/// Returns the floating-point math mode that will be used by default
/// for all subsequently created primitives.
///
/// @param mode Output FP math mode.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_get_default_fpmath_mode(dnnl_fpmath_mode_t *mode);

/// Sets the floating-point math mode that will be used by default
/// for all subsequently created primitives.
///
/// @param mode FP math mode. The possible values are:
///     #dnnl_fpmath_mode_strict,
///     #dnnl_fpmath_mode_bf16,
///     #dnnl_fpmath_mode_f16,
///     #dnnl_fpmath_mode_tf32,
///     #dnnl_fpmath_mode_any.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_set_default_fpmath_mode(dnnl_fpmath_mode_t mode);

/// @} dnnl_api_fpmath_mode

/// @addtogroup dnnl_api_service
/// @{

/// Configures verbose output to stdout.
///
/// @note
///     Enabling verbose output affects performance.
///     This setting overrides the ONEDNN_VERBOSE environment variable.
///
/// @param level Verbosity level:
///  - 0: no verbose output (default),
///  - 1: primitive and graph information at execution,
///  - 2: primitive and graph information at creation/compilation and execution.
/// @returns #dnnl_invalid_arguments/#dnnl::status::invalid_arguments if the
///     @p level value is invalid, and #dnnl_success/#dnnl::status::success on
///     success.
dnnl_status_t DNNL_API dnnl_set_verbose(int level);

/// Returns library version information.
/// @returns Pointer to a constant structure containing
///  - major: major version number,
///  - minor: minor version number,
///  - patch: patch release number,
///  - hash: git commit hash.
const dnnl_version_t DNNL_API *dnnl_version(void);

/// @} dnnl_api_service

/// @} dnnl_api_common

/// @} dnnl_api

#ifdef __cplusplus
}
#endif

#endif /* ONEAPI_DNNL_DNNL_COMMON_H */
