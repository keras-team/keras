// Copyright 2016 The Gemmlowp Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef GEMMLOWP_META_QUANTIZED_MUL_KERNELS_H_
#define GEMMLOWP_META_QUANTIZED_MUL_KERNELS_H_

#include <iostream>
#include <typeinfo>

#include "base.h"
#include "streams.h"

namespace gemmlowp {
namespace meta {

struct QuantizedStaticPreprocessed {
 public:
  int multiplicative_offset;
  int rounding_offset;
  int shift;
  int count;
};

template <typename InType, typename OutType, int m, int n, int k>
class MulKernel<InType, OutType, QuantizedStaticPreprocessed, RowMajor, m, n,
                k> {
 public:
  typedef FusedKernelParams<QuantizedStaticPreprocessed, RowMajor> FusedKernel;

  static void Multiply(const InType* lhs, const InType*,
                       const FusedKernel& params, OutType* result) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
    std::cout << "MulQSPR(" << typeid(InType).name() << ", "
              << typeid(OutType).name() << ")::Multiply() -- " << m << "x" << n
              << "x" << k << std::endl;
#endif
#else
    if (m != 0 && n != 0) {
      std::cerr << "FATAL: QuantizedStaticPreprocessed_RowMajor::Multiply not "
                << "implemented." << std::endl;
      std::exit(1);
    }
#endif
  }

#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  static void Debug(const FusedKernel& params) {
    std::cout << "MulQSPR(" << typeid(InType).name() << ", "
              << typeid(OutType).name() << ") -- " << m << "x" << n << "x" << k
              << std::endl;
    std::cout << "  params:" << std::endl;
    std::cout << "    kernel.multiplicative_offset: "
              << params.kernel.multiplicative_offset << std::endl;
    std::cout << "    kernel.rounding_offset: " << params.kernel.rounding_offset
              << std::endl;
    std::cout << "    kernel.shift: " << params.kernel.shift << std::endl;
    std::cout << "    kernel.count: " << params.kernel.count << std::endl;
    std::cout << "    output_stream.stride: " << params.output_stream.stride
              << std::endl;
  }
#endif
#endif
};

struct QuantizedStaticPreprocessedAsInt32 {
 public:
  int count;
};

template <typename InType, typename OutType, int m, int n, int k>
class MulKernel<InType, OutType, QuantizedStaticPreprocessedAsInt32, RowMajor,
                m, n, k> {
 public:
  typedef FusedKernelParams<QuantizedStaticPreprocessedAsInt32, RowMajor>
      FusedKernel;

  static void Multiply(const InType* lhs, const InType*,
                       const FusedKernel& params, OutType* result) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
    std::cout << "MulQSPI32R(" << typeid(InType).name() << ", "
              << typeid(OutType).name() << ")::Multiply() -- " << m << "x" << n
              << "x" << k << std::endl;
#endif
#else
    if (m != 0 && n != 0) {
      std::cerr << "FATAL: QuantizedStaticPreprocessedAsInt32_RowMajor::"
                << "Multiply not implemented." << std::endl;
      std::exit(1);
    }
#endif
  }

#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  static void Debug(const FusedKernel& params) {
    std::cout << "MulQSPI32R(" << typeid(InType).name() << ", "
              << typeid(OutType).name() << ") -- " << m << "x" << n << "x" << k
              << std::endl;
    std::cout << "  params:" << std::endl;
    std::cout << "    kernel.count: " << params.kernel.count << std::endl;
    std::cout << "    output_stream.stride: " << params.output_stream.stride
              << std::endl;
  }
#endif
#endif
};

struct QuantizedStaticPreprocessedAsFloat {
 public:
  int count;
  float scale;
};

template <typename InType, typename OutType, int m, int n, int k>
class MulKernel<InType, OutType, QuantizedStaticPreprocessedAsFloat, RowMajor,
                m, n, k> {
 public:
  typedef FusedKernelParams<QuantizedStaticPreprocessedAsFloat, RowMajor>
      FusedKernel;

  static void Multiply(const InType* lhs, const InType*,
                       const FusedKernel& params, OutType* result) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
    std::cout << "MulQSPFR(" << typeid(InType).name() << ", "
              << typeid(OutType).name() << ")::Multiply() -- " << m << "x" << n
              << "x" << k << std::endl;
#endif
#else
    if (m != 0 && n != 0) {
      std::cerr << "FATAL: QuantizedStaticPreprocessedAsFloat_RowMajor::"
                << "Multiply not implemented." << std::endl;
      std::exit(1);
    }
#endif
  }

#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  static void Debug(const FusedKernel& params) {
    std::cout << "MulQSPFR(" << typeid(InType).name() << ", "
              << typeid(OutType).name() << ") -- " << m << "x" << n << "x" << k
              << std::endl;
    std::cout << "  params:" << std::endl;
    std::cout << "    kernel.count: " << params.kernel.count << std::endl;
    std::cout << "    kernel.scale: " << params.kernel.scale << std::endl;
    std::cout << "    output_stream.stride: " << params.output_stream.stride
              << std::endl;
  }
#endif
#endif
};

}  // namespace meta
}  // namespace gemmlowp

#ifdef GEMMLOWP_NEON_32
#include "quantized_mul_kernels_arm_32.h"
#elif defined(GEMMLOWP_NEON_64)
#include "quantized_mul_kernels_arm_64.h"
#endif

#endif  // GEMMLOWP_META_QUANTIZED_MUL_KERNELS_H_
