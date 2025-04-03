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

// Decoded diagnostic abstraction
//
// This file declares DecodedDiagnostic.

#ifndef TFRT_HOST_CONTEXT_DIAGNOSTIC_H_
#define TFRT_HOST_CONTEXT_DIAGNOSTIC_H_

#include <optional>
#include <string_view>
#include <utility>

#include "absl/status/status.h"  // from @com_google_absl
#include "tfrt/host_context/async_value.h"
#include "tfrt/host_context/location.h"
#include "tfrt/support/forward_decls.h"
#include "tfrt/support/string_util.h"

namespace tfrt {

class ExecutionContext;

// This is a simple representation of a decoded diagnostic.
struct DecodedDiagnostic {
  // TODO(ezhulenev): Remove this constructor, as the DecodedDiagnostic should
  // only be used when location is important. If location is not needed, then
  // passing absl::Status should be a preferred option.
  explicit DecodedDiagnostic(absl::Status status) : status(std::move(status)) {
    assert(!this->status.ok() && "must be non-ok status");
  }

  DecodedDiagnostic(DecodedLocation location, absl::Status status)
      : location(std::move(location)), status(std::move(status)) {
    assert(!this->status.ok() && "must be non-ok status");
  }

  std::string_view message() const { return status.message(); }

  absl::StatusCode code() const { return status.code(); }

  std::optional<DecodedLocation> location;
  absl::Status status;
};

raw_ostream& operator<<(raw_ostream& os, const DecodedDiagnostic& diagnostic);

DecodedDiagnostic EmitError(const ExecutionContext& exec_ctx,
                            absl::Status status);

template <typename... Args>
DecodedDiagnostic EmitError(const ExecutionContext& exec_ctx, Args&&... args) {
  return EmitError(exec_ctx,
                   absl::InternalError(StrCat(std::forward<Args>(args)...)));
}

// For consistency, the error message should start with a lower case letter
// and not end with a period.
RCReference<ErrorAsyncValue> EmitErrorAsync(const ExecutionContext& exec_ctx,
                                            absl::Status status);

RCReference<ErrorAsyncValue> EmitErrorAsync(const ExecutionContext& exec_ctx,
                                            std::string_view message);

RCReference<ErrorAsyncValue> EmitErrorAsync(const ExecutionContext& exec_ctx,
                                            Error error);

}  // namespace tfrt

#endif  // TFRT_HOST_CONTEXT_DIAGNOSTIC_H_
