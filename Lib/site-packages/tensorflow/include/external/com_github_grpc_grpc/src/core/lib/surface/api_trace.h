/*
 *
 * Copyright 2015 gRPC authors.
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

#ifndef GRPC_CORE_LIB_SURFACE_API_TRACE_H
#define GRPC_CORE_LIB_SURFACE_API_TRACE_H

#include <grpc/support/port_platform.h>

#include <grpc/support/log.h>
#include "src/core/lib/debug/trace.h"

extern grpc_core::TraceFlag grpc_api_trace;

/* Provide unwrapping macros because we're in C89 and variadic macros weren't
   introduced until C99... */
#define GRPC_API_TRACE_UNWRAP0()
#define GRPC_API_TRACE_UNWRAP1(a) , a
#define GRPC_API_TRACE_UNWRAP2(a, b) , a, b
#define GRPC_API_TRACE_UNWRAP3(a, b, c) , a, b, c
#define GRPC_API_TRACE_UNWRAP4(a, b, c, d) , a, b, c, d
#define GRPC_API_TRACE_UNWRAP5(a, b, c, d, e) , a, b, c, d, e
#define GRPC_API_TRACE_UNWRAP6(a, b, c, d, e, f) , a, b, c, d, e, f
#define GRPC_API_TRACE_UNWRAP7(a, b, c, d, e, f, g) , a, b, c, d, e, f, g
#define GRPC_API_TRACE_UNWRAP8(a, b, c, d, e, f, g, h) , a, b, c, d, e, f, g, h
#define GRPC_API_TRACE_UNWRAP9(a, b, c, d, e, f, g, h, i) \
  , a, b, c, d, e, f, g, h, i
#define GRPC_API_TRACE_UNWRAP10(a, b, c, d, e, f, g, h, i, j) \
  , a, b, c, d, e, f, g, h, i, j

/* Due to the limitations of C89's preprocessor, the arity of the var-arg list
   'nargs' must be specified. */
#define GRPC_API_TRACE(fmt, nargs, args)                      \
  if (GRPC_TRACE_FLAG_ENABLED(grpc_api_trace)) {              \
    gpr_log(GPR_INFO, fmt GRPC_API_TRACE_UNWRAP##nargs args); \
  }

#endif /* GRPC_CORE_LIB_SURFACE_API_TRACE_H */
