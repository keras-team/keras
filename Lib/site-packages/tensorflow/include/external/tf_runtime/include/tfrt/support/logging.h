/*
 * Copyright 2020 The TensorFlow Runtime Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// This file declares the logging library.

#ifndef TFRT_SUPPORT_LOGGING_H_
#define TFRT_SUPPORT_LOGGING_H_

#include "llvm/Support/raw_ostream.h"

// There are four severity levels: INFO, WARNING, ERROR, FATAL.
// There are several variants: LOG and LOG_IF or DLOG and DLOG_IF.
// LOG and LOG_IF are always on. DLOG and DLOG_IF are compiled away for
// optimized build.
// The recommendation is to use DLOG and DLOG_IF only in core TFRT code because
// as a library, internal logging is not expected to be exposed for release.
// Kernel implementations are free to use either LOG or DLOG.

namespace tfrt {

enum class Severity : int8_t {
  INFO = 0,
  WARNING = 1,
  ERROR = 2,
  FATAL = 3,
};

namespace internal {
class LogStream : public llvm::raw_ostream {
 public:
  LogStream(const char* fname, int line, Severity severity);
  ~LogStream() override;

 private:
  void write_impl(const char* Ptr, size_t Size) override;
  uint64_t current_pos() const override;

  uint64_t pos_;
  Severity severity_;
};

// LogStreamFatal marks the destructor as noreturn to allow code paths with
// LOG(FATAL) to not return any value.
class LogStreamFatal : public LogStream {
 public:
  LogStreamFatal(const char* file, int line);
  [[noreturn]] ~LogStreamFatal() = default;
};

// Uses the lower operator & precedence to voidify a LogStream reference, so
// that the ternary VLOG() implementation is balanced, type wise.
struct LogStreamVoidifier {
  template <typename T>
  void operator&(const T&) const {}
};
}  // namespace internal

#define TFRT_LOG_INFO \
  ::tfrt::internal::LogStream(__FILE__, __LINE__, ::tfrt::Severity::INFO)
#define TFRT_LOG_WARNING \
  ::tfrt::internal::LogStream(__FILE__, __LINE__, ::tfrt::Severity::WARNING)
#define TFRT_LOG_ERROR \
  ::tfrt::internal::LogStream(__FILE__, __LINE__, ::tfrt::Severity::ERROR)
#define TFRT_LOG_FATAL ::tfrt::internal::LogStreamFatal(__FILE__, __LINE__)

// `TFRT_LOG` entry point.
#define TFRT_LOG(severity) \
  switch (0)               \
  default:                 \
    TFRT_LOG_##severity

// `TFRT_INTERNAL_LOG_CONDITION` prefixes another macro that expands to a
// temporary `LogStream` instantiation.
#define TFRT_INTERNAL_LOG_CONDITION(condition) \
  !(condition) ? (void)0 : ::tfrt::internal::LogStreamVoidifier()&

// `TFRT_DLOG` behaves like `TFRT_LOG` in debug mode (i.e. `#ifndef NDEBUG`).
// Otherwise, it compiles away and does nothing. Note that `TFRT_DLOG(FATAL)`
// does not terminate the program if `NDEBUG` is defined.
#ifndef NDEBUG
#define TFRT_DLOG TFRT_LOG
#else
#define TFRT_DLOG(severity) \
  switch (0)                \
  default:                  \
    TFRT_INTERNAL_LOG_CONDITION(false) TFRT_LOG_##severity
#endif

// TFRT_LOG_IF and friends add a second argument which specifies a condition. If
// the condition is false, nothing is logged.
#define TFRT_LOG_IF(severity, condition) \
  switch (0)                             \
  default:                               \
    TFRT_INTERNAL_LOG_CONDITION(condition) TFRT_LOG_##severity

#ifndef NDEBUG
#define TFRT_DLOG_IF TFRT_LOG_IF
#else
#define TFRT_DLOG_IF(severity, condition) \
  switch (0)                              \
  default:                                \
    TFRT_INTERNAL_LOG_CONDITION(false && condition) TFRT_LOG_##severity
#endif

}  // namespace tfrt

#endif  // TFRT_SUPPORT_LOGGING_H_
