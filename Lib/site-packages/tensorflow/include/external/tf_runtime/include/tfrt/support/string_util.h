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

// This file introduces utility functions related to string manipulation.

#ifndef TFRT_SUPPORT_STRING_UTIL_H_
#define TFRT_SUPPORT_STRING_UTIL_H_

#include <sstream>
#include <string>
#include <utility>

#include "llvm/Support/raw_ostream.h"

namespace tfrt {

namespace internal {

template <typename StreamT>
inline void ToStreamHelper(StreamT& os) {}

template <typename StreamT, typename T, typename... Args>
void ToStreamHelper(StreamT& os, T&& v, Args&&... args) {
  os << std::forward<T>(v);
  ToStreamHelper(os, std::forward<Args>(args)...);
}

}  // namespace internal

// Utility function to stream arguments into a std::string.

template <typename... Args>
std::string StrCat(Args&&... args) {
  std::string str;
  llvm::raw_string_ostream sstr(str);
  internal::ToStreamHelper(sstr, std::forward<Args>(args)...);
  sstr.flush();
  return str;
}

template <typename... Args>
std::string OstreamStrCat(Args&&... args) {
  std::ostringstream sstr;
  internal::ToStreamHelper(sstr, std::forward<Args>(args)...);
  return sstr.str();
}

// Utility function to append arguments after `str` to `str`.
template <typename... Args>
void StrAppend(std::string* str, Args&&... args) {
  llvm::raw_string_ostream sstr(*str);
  internal::ToStreamHelper(sstr, std::forward<Args>(args)...);
}

template <typename IteratorT>
std::string Join(IteratorT begin, IteratorT end, llvm::StringRef separator) {
  if (begin == end) return "";

  std::string str;
  llvm::raw_string_ostream os(str);

  os << (*begin);
  while (++begin != end) {
    os << separator;
    os << (*begin);
  }
  os.flush();
  return str;
}

template <typename Range>
std::string Join(const Range& range, llvm::StringRef separator) {
  return Join(range.begin(), range.end(), separator);
}

// Converts from an int64 to a human readable string representing the
// same number, using decimal powers.  e.g. 1200000 -> "1.20M".
std::string HumanReadableNum(int64_t value);

// Converts from an int64 representing a number of bytes to a
// human readable string representing the same number.
// e.g. 12345678 -> "11.77MiB".
std::string HumanReadableNumBytes(int64_t num_bytes);

// Converts a time interval as double to a human readable
// string. For example:
//   0.001       -> "1 ms"
//   10.0        -> "10 s"
//   933120.0    -> "10.8 days"
//   39420000.0  -> "1.25 years"
//   -10         -> "-10 s"
std::string HumanReadableElapsedTime(double seconds);

}  // namespace tfrt

#endif  // TFRT_SUPPORT_STRING_UTIL_H_
