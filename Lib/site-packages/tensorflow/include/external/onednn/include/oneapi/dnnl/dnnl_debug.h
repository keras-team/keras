/*******************************************************************************
* Copyright 2018-2024 Intel Corporation
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

// DO NOT EDIT, AUTO-GENERATED
// Use this script to update the file: scripts/generate_dnnl_debug.py

// clang-format off

#ifndef ONEAPI_DNNL_DNNL_DEBUG_H
#define ONEAPI_DNNL_DNNL_DEBUG_H

/// @file
/// Debug capabilities

#include "oneapi/dnnl/dnnl_config.h"
#include "oneapi/dnnl/dnnl_types.h"

#ifdef __cplusplus
extern "C" {
#endif

const char DNNL_API *dnnl_status2str(dnnl_status_t v);
const char DNNL_API *dnnl_dt2str(dnnl_data_type_t v);
const char DNNL_API *dnnl_fpmath_mode2str(dnnl_fpmath_mode_t v);
const char DNNL_API *dnnl_accumulation_mode2str(dnnl_accumulation_mode_t v);
const char DNNL_API *dnnl_engine_kind2str(dnnl_engine_kind_t v);
#ifdef DNNL_EXPERIMENTAL_SPARSE
const char DNNL_API *dnnl_sparse_encoding2str(dnnl_sparse_encoding_t v);
#endif
const char DNNL_API *dnnl_fmt_tag2str(dnnl_format_tag_t v);
const char DNNL_API *dnnl_prop_kind2str(dnnl_prop_kind_t v);
const char DNNL_API *dnnl_prim_kind2str(dnnl_primitive_kind_t v);
const char DNNL_API *dnnl_alg_kind2str(dnnl_alg_kind_t v);
const char DNNL_API *dnnl_rnn_flags2str(dnnl_rnn_flags_t v);
const char DNNL_API *dnnl_rnn_direction2str(dnnl_rnn_direction_t v);
const char DNNL_API *dnnl_scratchpad_mode2str(dnnl_scratchpad_mode_t v);
const char DNNL_API *dnnl_cpu_isa2str(dnnl_cpu_isa_t v);
const char DNNL_API *dnnl_cpu_isa_hints2str(dnnl_cpu_isa_hints_t v);

const char DNNL_API *dnnl_runtime2str(unsigned v);
const char DNNL_API *dnnl_fmt_kind2str(dnnl_format_kind_t v);

#ifdef __cplusplus
}
#endif

#endif
