/*
    Copyright (c) 2005-2024 Intel Corporation

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
*/

#ifndef __TBB_version_H
#define __TBB_version_H

// Exclude all includes during .rc files compilation
#ifndef RC_INVOKED
    #include "detail/_config.h"
    #include "detail/_namespace_injection.h"
#else
    #define __TBB_STRING_AUX(x) #x
    #define __TBB_STRING(x) __TBB_STRING_AUX(x)
#endif

// Product version
#define TBB_VERSION_MAJOR 2021
// Update version
#define TBB_VERSION_MINOR 2
// "Patch" version for custom releases
#define TBB_VERSION_PATCH 5
// Suffix string
#define __TBB_VERSION_SUFFIX ""
// Full official version string
#define TBB_VERSION_STRING              \
    __TBB_STRING(TBB_VERSION_MAJOR) "." \
    __TBB_STRING(TBB_VERSION_MINOR) "." \
    __TBB_STRING(TBB_VERSION_PATCH)     \
    __TBB_VERSION_SUFFIX

// OneAPI oneTBB specification version
#define ONETBB_SPEC_VERSION "1.0"
// Full interface version
#define TBB_INTERFACE_VERSION 12025
// Major interface version
#define TBB_INTERFACE_VERSION_MAJOR (TBB_INTERFACE_VERSION/1000)
// Minor interface version
#define TBB_INTERFACE_VERSION_MINOR (TBB_INTERFACE_VERSION%1000/10)

// The binary compatibility version
// To be used in SONAME, manifests, etc.
#define __TBB_BINARY_VERSION 12

//! TBB_VERSION support
#ifndef ENDL
#define ENDL "\n"
#endif

//TBB_REVAMP_TODO: consider enabling version_string.ver generation
//TBB_REVAMP_TODO: #include "version_string.ver"

#define __TBB_ONETBB_SPEC_VERSION(N) #N ": SPECIFICATION VERSION\t" ONETBB_SPEC_VERSION ENDL
#define __TBB_VERSION_NUMBER(N) #N ": VERSION\t\t" TBB_VERSION_STRING ENDL
#define __TBB_INTERFACE_VERSION_NUMBER(N) #N ": INTERFACE VERSION\t" __TBB_STRING(TBB_INTERFACE_VERSION) ENDL

#ifndef TBB_USE_DEBUG
    #define __TBB_VERSION_USE_DEBUG(N) #N ": TBB_USE_DEBUG\tundefined" ENDL
#elif TBB_USE_DEBUG==0
    #define __TBB_VERSION_USE_DEBUG(N) #N ": TBB_USE_DEBUG\t0" ENDL
#elif TBB_USE_DEBUG==1
    #define __TBB_VERSION_USE_DEBUG(N) #N ": TBB_USE_DEBUG\t1" ENDL
#elif TBB_USE_DEBUG==2
    #define __TBB_VERSION_USE_DEBUG(N) #N ": TBB_USE_DEBUG\t2" ENDL
#else
    #error Unexpected value for TBB_USE_DEBUG
#endif

#ifndef TBB_USE_ASSERT
    #define __TBB_VERSION_USE_ASSERT(N) #N ": TBB_USE_ASSERT\tundefined" ENDL
#elif TBB_USE_ASSERT==0
    #define __TBB_VERSION_USE_ASSERT(N) #N ": TBB_USE_ASSERT\t0" ENDL
#elif TBB_USE_ASSERT==1
    #define __TBB_VERSION_USE_ASSERT(N) #N ": TBB_USE_ASSERT\t1" ENDL
#elif TBB_USE_ASSERT==2
    #define __TBB_VERSION_USE_ASSERT(N) #N ": TBB_USE_ASSERT\t2" ENDL
#else
    #error Unexpected value for TBB_USE_ASSERT
#endif

#define TBB_VERSION_STRINGS_P(N)                \
    __TBB_ONETBB_SPEC_VERSION(N)                \
    __TBB_VERSION_NUMBER(N)                     \
    __TBB_INTERFACE_VERSION_NUMBER(N)           \
    __TBB_VERSION_USE_DEBUG(N)                  \
    __TBB_VERSION_USE_ASSERT(N)

#define TBB_VERSION_STRINGS TBB_VERSION_STRINGS_P(oneTBB)
#define TBBMALLOC_VERSION_STRINGS TBB_VERSION_STRINGS_P(TBBmalloc)

//! The function returns the version string for the Intel(R) oneAPI Threading Building Blocks (oneTBB)
//! shared library being used.
/**
 * The returned pointer is an address of a string in the shared library.
 * It can be different than the TBB_VERSION_STRING obtained at compile time.
 */
extern "C" const char* __TBB_EXPORTED_FUNC TBB_runtime_version();

//! The function returns the interface version of the oneTBB shared library being used.
/**
 * The returned version is determined at runtime, not at compile/link time.
 * It can be different than the value of TBB_INTERFACE_VERSION obtained at compile time.
 */
extern "C" int __TBB_EXPORTED_FUNC TBB_runtime_interface_version();

#endif // __TBB_version_H
