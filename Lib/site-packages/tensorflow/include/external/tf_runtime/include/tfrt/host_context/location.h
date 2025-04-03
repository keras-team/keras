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

// This file declares some routines for working with diagnostics: it declares
// the Location class, which is the primary encoded form of a location, as well
// as the LocationHandler class which decodes them.

#ifndef TFRT_HOST_CONTEXT_LOCATION_H_
#define TFRT_HOST_CONTEXT_LOCATION_H_

#include <cstdint>
#include <optional>
#include <string>

#include "tfrt/support/forward_decls.h"
#include "tfrt/support/variant.h"

namespace tfrt {

class LocationHandler;

struct FileLineColLocation {
  std::string filename;
  int line = -1;
  int column = -1;
};

struct OpaqueLocation {
  std::string loc;
};

struct DebugInfo {
  std::string info;
};

using DecodedLocation = Variant<FileLineColLocation, OpaqueLocation>;

// This is an opaque location token that is passed to kernel implementations,
// and is used to report errors. It should remain a simple POD type.
class Location {
 public:
  Location() = default;
  Location(const LocationHandler* handler, intptr_t data)
      : data(data), handler_(handler) {}

  // Location has an optional semantic. A default constructed evaluates to
  // false.
  explicit operator bool() const { return handler_ != nullptr; }

  DecodedLocation Decode() const;

  std::optional<DebugInfo> GetDebugInfo() const;

  const LocationHandler* GetHandler() const { return handler_; }

  // Opaque implementation details of this location, only interpretable by the
  // location handler.
  intptr_t data = 0;

 private:
  friend class LocationHandler;
  const LocationHandler* handler_ = nullptr;
};

// This is a virtual base class used by things that create locations.
class LocationHandler {
 public:
  virtual ~LocationHandler();

  virtual DecodedLocation DecodeLocation(Location loc) const = 0;
  virtual std::optional<DebugInfo> GetDebugInfo(Location /*loc*/) const {
    return std::nullopt;
  }
};

inline DecodedLocation Location::Decode() const {
  if (!handler_) return DecodedLocation();
  return handler_->DecodeLocation(*this);
}

inline std::optional<DebugInfo> Location::GetDebugInfo() const {
  if (!handler_) return std::nullopt;
  return handler_->GetDebugInfo(*this);
}

raw_ostream& operator<<(raw_ostream& os, const FileLineColLocation& loc);
raw_ostream& operator<<(raw_ostream& os, const OpaqueLocation& loc);
raw_ostream& operator<<(raw_ostream& os, const DecodedLocation& loc);

}  // namespace tfrt

#endif  // TFRT_HOST_CONTEXT_LOCATION_H_
