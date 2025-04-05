/*
 *
 * Copyright 2019 gRPC authors.
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
 *
 */

#ifndef GRPC_CORE_LIB_GPRPP_GLOBAL_CONFIG_H
#define GRPC_CORE_LIB_GPRPP_GLOBAL_CONFIG_H

#include <grpc/support/port_platform.h>

#include <stdint.h>

// --------------------------------------------------------------------
// How to use global configuration variables:
//
// Defining config variables of a specified type:
//   GPR_GLOBAL_CONFIG_DEFINE_*TYPE*(name, default_value, help);
//
// Supported TYPEs: BOOL, INT32, STRING
//
// It's recommended to use lowercase letters for 'name' like
// regular variables. The builtin configuration system uses
// environment variable and the name is converted to uppercase
// when looking up the value. For example,
// GPR_GLOBAL_CONFIG_DEFINE(grpc_latency) looks up the value with the
// name, "GRPC_LATENCY".
//
// The variable initially has the specified 'default_value'
// which must be an expression convertible to 'Type'.
// 'default_value' may be evaluated 0 or more times,
// and at an unspecified time; keep it
// simple and usually free of side-effects.
//
// GPR_GLOBAL_CONFIG_DEFINE_*TYPE* should not be called in a C++ header.
// It should be called at the top-level (outside any namespaces)
// in a .cc file.
//
// Getting the variables:
//   GPR_GLOBAL_CONFIG_GET(name)
//
// If error happens during getting variables, error messages will
// be logged and default value will be returned.
//
// Setting the variables with new value:
//   GPR_GLOBAL_CONFIG_SET(name, new_value)
//
// Declaring config variables for other modules to access:
//   GPR_GLOBAL_CONFIG_DECLARE_*TYPE*(name)
//
// * Caveat for setting global configs at runtime
//
// Setting global configs at runtime multiple times is safe but it doesn't
// mean that it will have a valid effect on the module depending configs.
// In unit tests, it may be unpredictable to set different global configs
// between test cases because grpc init and shutdown can ignore changes.
// It's considered safe to set global configs before the first call to
// grpc_init().

// --------------------------------------------------------------------
// How to customize the global configuration system:
//
// How to read and write configuration value can be customized.
// Builtin system uses environment variables but it can be extended to
// support command-line flag, file, etc.
//
// To customize it, following macros should be redefined.
//
//   GPR_GLOBAL_CONFIG_DEFINE_BOOL
//   GPR_GLOBAL_CONFIG_DEFINE_INT32
//   GPR_GLOBAL_CONFIG_DEFINE_STRING
//
// These macros should define functions for getting and setting variable.
// For example, GPR_GLOBAL_CONFIG_DEFINE_BOOL(test, ...) would define two
// functions.
//
//   bool gpr_global_config_get_test();
//   void gpr_global_config_set_test(bool value);

#include "src/core/lib/gprpp/global_config_env.h"

#include "src/core/lib/gprpp/global_config_custom.h"

#endif /* GRPC_CORE_LIB_GPRPP_GLOBAL_CONFIG_H */
