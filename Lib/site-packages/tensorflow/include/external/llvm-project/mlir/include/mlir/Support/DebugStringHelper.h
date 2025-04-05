//===- DebugStringHelper.h - helpers to generate debug strings --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Convenience functions to make it easier to get a string representation for
// ops that have a print method. For use in debugging output and errors
// returned.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_SUPPORT_DEBUGSTRINGHELPER_H
#define MLIR_SUPPORT_DEBUGSTRINGHELPER_H

#include <string>

#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_os_ostream.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {

// Simple helper function that returns a string as printed from a op.
template <typename T>
static std::string debugString(T &&op) {
  std::string instrStr;
  llvm::raw_string_ostream os(instrStr);
  os << op;
  return os.str();
}

} // namespace mlir

inline std::ostream &operator<<(std::ostream &out, const llvm::Twine &twine) {
  llvm::raw_os_ostream rout(out);
  rout << twine;
  return out;
}

#endif // MLIR_SUPPORT_DEBUGSTRINGHELPER_H
