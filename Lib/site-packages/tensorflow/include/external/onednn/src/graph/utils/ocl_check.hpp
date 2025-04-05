/*******************************************************************************
* Copyright 2024 Intel Corporation
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

#ifndef GRAPH_UTILS_OCL_CHECK_HPP
#define GRAPH_UTILS_OCL_CHECK_HPP

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
#if __has_include(<CL/cl.h>)
#include <CL/cl.h>
#else
#error "Unsupported compiler"
#endif

#if __has_include(<CL/cl_ext.h>)
#include <CL/cl_ext.h>
#else
#error "Unsupported compiler"
#endif
#endif

#include "gpu/intel/ocl/ocl_utils.hpp"

#endif
