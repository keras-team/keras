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

#ifndef GRPC_IMPL_CODEGEN_LOG_H
#define GRPC_IMPL_CODEGEN_LOG_H

#include <grpc/impl/codegen/port_platform.h>

#include <stdarg.h>
#include <stdlib.h> /* for abort() */

#ifdef __cplusplus
extern "C" {
#endif

/** GPR log API.

   Usage (within grpc):

   int argument1 = 3;
   char* argument2 = "hello";
   gpr_log(GPR_DEBUG, "format string %d", argument1);
   gpr_log(GPR_INFO, "hello world");
   gpr_log(GPR_ERROR, "%d %s!!", argument1, argument2); */

/** The severity of a log message - use the #defines below when calling into
   gpr_log to additionally supply file and line data */
typedef enum gpr_log_severity {
  GPR_LOG_SEVERITY_DEBUG,
  GPR_LOG_SEVERITY_INFO,
  GPR_LOG_SEVERITY_ERROR
} gpr_log_severity;

#define GPR_LOG_VERBOSITY_UNSET -1

/** Returns a string representation of the log severity */
GPRAPI const char* gpr_log_severity_string(gpr_log_severity severity);

/** Macros to build log contexts at various severity levels */
#define GPR_DEBUG __FILE__, __LINE__, GPR_LOG_SEVERITY_DEBUG
#define GPR_INFO __FILE__, __LINE__, GPR_LOG_SEVERITY_INFO
#define GPR_ERROR __FILE__, __LINE__, GPR_LOG_SEVERITY_ERROR

/** Log a message. It's advised to use GPR_xxx above to generate the context
 * for each message */
GPRAPI void gpr_log(const char* file, int line, gpr_log_severity severity,
                    const char* format, ...) GPR_PRINT_FORMAT_CHECK(4, 5);

GPRAPI int gpr_should_log(gpr_log_severity severity);

GPRAPI void gpr_log_message(const char* file, int line,
                            gpr_log_severity severity, const char* message);

/** Set global log verbosity */
GPRAPI void gpr_set_log_verbosity(gpr_log_severity min_severity_to_print);

GPRAPI void gpr_log_verbosity_init(void);

/** Log overrides: applications can use this API to intercept logging calls
   and use their own implementations */

struct gpr_log_func_args {
  const char* file;
  int line;
  gpr_log_severity severity;
  const char* message;
};

typedef struct gpr_log_func_args gpr_log_func_args;

typedef void (*gpr_log_func)(gpr_log_func_args* args);
GPRAPI void gpr_set_log_function(gpr_log_func func);

/** abort() the process if x is zero, having written a line to the log.

   Intended for internal invariants.  If the error can be recovered from,
   without the possibility of corruption, or might best be reflected via
   an exception in a higher-level language, consider returning error code.  */
#define GPR_ASSERT(x)                                 \
  do {                                                \
    if (GPR_UNLIKELY(!(x))) {                         \
      gpr_log(GPR_ERROR, "assertion failed: %s", #x); \
      abort();                                        \
    }                                                 \
  } while (0)

#ifndef NDEBUG
#define GPR_DEBUG_ASSERT(x) GPR_ASSERT(x)
#else
#define GPR_DEBUG_ASSERT(x)
#endif

#ifdef __cplusplus
}
#endif

#endif /* GRPC_IMPL_CODEGEN_LOG_H */
