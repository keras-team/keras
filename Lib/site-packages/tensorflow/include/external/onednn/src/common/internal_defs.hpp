/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#ifndef COMMON_INTERNAL_DEFS_HPP
#define COMMON_INTERNAL_DEFS_HPP

#if defined(DNNL_DLL)
#define DNNL_WEAK DNNL_HELPER_DLL_EXPORT
#else
#if defined(__GNUC__) || defined(__clang__)
#define DNNL_WEAK __attribute__((weak))
#else
#define DNNL_WEAK
#endif
#endif

#if defined(DNNL_DLL)
#define DNNL_STRONG DNNL_HELPER_DLL_EXPORT
#else
#define DNNL_STRONG
#endif

#endif
