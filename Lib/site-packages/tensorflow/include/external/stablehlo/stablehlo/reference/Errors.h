/* Copyright 2022 The StableHLO Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef STABLEHLO_REFERENCE_ERRORS_H
#define STABLEHLO_REFERENCE_ERRORS_H

#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"

namespace mlir {
namespace stablehlo {

/// Wrapper error handing function for StableHLO. Creates an invalid argument
/// error using a format string and a variadic number of arguments to the format
/// string.
template <typename... Ts>
inline llvm::Error invalidArgument(char const *Fmt, const Ts &...Vals) {
  return createStringError(llvm::errc::invalid_argument, Fmt, Vals...);
}

/// Wrapper error handing function for StableHLO. Creates an invalid argument
/// error using the specified function name and fallback name as error text.
llvm::Error wrapFallbackStatus(llvm::Error status, llvm::StringRef funcName,
                               llvm::StringRef fallbackName);

}  // namespace stablehlo
}  // namespace mlir

#endif  // STABLEHLO_REFERENCE_ERRORS_H
