//===- MemRefDescriptor.h - MemRef descriptor constants ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines constants that are used in LLVM dialect equivalents of MemRef type.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_LIB_CONVERSION_LLVMCOMMON_MEMREFDESCRIPTOR_H
#define MLIR_LIB_CONVERSION_LLVMCOMMON_MEMREFDESCRIPTOR_H

static constexpr unsigned kAllocatedPtrPosInMemRefDescriptor = 0;
static constexpr unsigned kAlignedPtrPosInMemRefDescriptor = 1;
static constexpr unsigned kOffsetPosInMemRefDescriptor = 2;
static constexpr unsigned kSizePosInMemRefDescriptor = 3;
static constexpr unsigned kStridePosInMemRefDescriptor = 4;

static constexpr unsigned kRankInUnrankedMemRefDescriptor = 0;
static constexpr unsigned kPtrInUnrankedMemRefDescriptor = 1;

#endif // MLIR_LIB_CONVERSION_LLVMCOMMON_MEMREFDESCRIPTOR_H
