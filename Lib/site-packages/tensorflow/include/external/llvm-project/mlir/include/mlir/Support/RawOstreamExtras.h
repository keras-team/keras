//===- RawOstreamExtras.h - Extensions to LLVM's raw_ostream.h --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

namespace llvm {
class raw_ostream;
} // namespace llvm

namespace mlir {
/// Returns a raw output stream that simply discards the output, but in a
/// thread-safe manner. Similar to llvm::nulls.
llvm::raw_ostream &thread_safe_nulls();
} // namespace mlir
