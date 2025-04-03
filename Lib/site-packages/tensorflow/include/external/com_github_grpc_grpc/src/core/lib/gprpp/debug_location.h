/*
 *
 * Copyright 2017 gRPC authors.
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

#ifndef GRPC_CORE_LIB_GPRPP_DEBUG_LOCATION_H
#define GRPC_CORE_LIB_GPRPP_DEBUG_LOCATION_H

namespace grpc_core {

// Used for tracking file and line where a call is made for debug builds.
// No-op for non-debug builds.
// Callers can use the DEBUG_LOCATION macro in either case.
#ifndef NDEBUG
// TODO(roth): See if there's a way to automatically populate this,
// similarly to how absl::SourceLocation::current() works, so that
// callers don't need to explicitly pass DEBUG_LOCATION anywhere.
class DebugLocation {
 public:
  DebugLocation(const char* file, int line) : file_(file), line_(line) {}
  const char* file() const { return file_; }
  int line() const { return line_; }

 private:
  const char* file_;
  const int line_;
};
#define DEBUG_LOCATION ::grpc_core::DebugLocation(__FILE__, __LINE__)
#else
class DebugLocation {
 public:
  const char* file() const { return nullptr; }
  int line() const { return -1; }
};
#define DEBUG_LOCATION ::grpc_core::DebugLocation()
#endif

}  // namespace grpc_core

#endif /* GRPC_CORE_LIB_GPRPP_DEBUG_LOCATION_H */
