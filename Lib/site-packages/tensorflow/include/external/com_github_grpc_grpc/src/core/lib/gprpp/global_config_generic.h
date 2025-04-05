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

#ifndef GRPC_CORE_LIB_GPRPP_GLOBAL_CONFIG_GENERIC_H
#define GRPC_CORE_LIB_GPRPP_GLOBAL_CONFIG_GENERIC_H

#include <grpc/support/port_platform.h>

#include "src/core/lib/gprpp/memory.h"

#include <stdint.h>

#define GPR_GLOBAL_CONFIG_GET(name) gpr_global_config_get_##name()

#define GPR_GLOBAL_CONFIG_SET(name, value) gpr_global_config_set_##name(value)

#define GPR_GLOBAL_CONFIG_DECLARE_BOOL(name)  \
  extern bool gpr_global_config_get_##name(); \
  extern void gpr_global_config_set_##name(bool value)

#define GPR_GLOBAL_CONFIG_DECLARE_INT32(name)    \
  extern int32_t gpr_global_config_get_##name(); \
  extern void gpr_global_config_set_##name(int32_t value)

#define GPR_GLOBAL_CONFIG_DECLARE_STRING(name)                      \
  extern grpc_core::UniquePtr<char> gpr_global_config_get_##name(); \
  extern void gpr_global_config_set_##name(const char* value)

#endif /* GRPC_CORE_LIB_GPRPP_GLOBAL_CONFIG_GENERIC_H */
