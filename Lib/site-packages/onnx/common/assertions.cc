// Copyright (c) ONNX Project Contributors

/*
 * SPDX-License-Identifier: Apache-2.0
 */

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#include "onnx/common/assertions.h"

#include <array>
#include <cstdarg>
#include <cstdio>

#include "onnx/common/common.h"

namespace ONNX_NAMESPACE {

std::string barf(const char* fmt, ...) {
  constexpr size_t buffer_size = 2048;
  std::array<char, buffer_size> msg{};
  va_list args;

  va_start(args, fmt);

  // use fixed length for buffer "msg" to avoid buffer overflow
  vsnprintf(static_cast<char*>(msg.data()), msg.size() - 1, fmt, args);

  // ensure null-terminated string to avoid out of bounds read
  msg.back() = '\0';
  va_end(args);

  return std::string(msg.data());
}

void throw_assert_error(std::string& msg) {
  ONNX_THROW_EX(assert_error(msg));
}

void throw_tensor_error(std::string& msg) {
  ONNX_THROW_EX(tensor_error(msg));
}

} // namespace ONNX_NAMESPACE
