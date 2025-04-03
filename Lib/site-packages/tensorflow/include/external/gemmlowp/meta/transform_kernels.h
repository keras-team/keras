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

#ifndef GEMMLOWP_META_TRANSFORM_KERNELS_H_
#define GEMMLOWP_META_TRANSFORM_KERNELS_H_

#include "base.h"

namespace gemmlowp {
namespace meta {

struct Quantize {
  float range_min;
  float range_offset;
  float range_scale;
  int count;
};

struct Dequantize {
  float range_min;
  float range_offset;
  float range_scale;
  int count;
};

struct Requantize {
  float input_range_min;
  float input_range_offset;
  float input_range_scale;
  float output_range_min;
  float output_range_offset;
  float one_over_output_range_scale;
  int count;
};

template <typename Type>
struct MinMax {
  Type min;
  Type max;
  int count;
};

template <typename BiasType>
struct BiasAdd {
  float input_range_min;
  float input_range_offset;
  float input_range_scale;
  float bias_range_min;
  float bias_range_offset;
  float bias_range_scale;
  float output_range_min;
  float output_range_offset;
  float one_over_output_range_scale;
  int count;
  int rows;
  const BiasType* bias;
};

template <typename InType, typename OutType, int kernel_size, int leftovers>
class Transform1DKernel<InType, OutType, Quantize, kernel_size, leftovers> {
 public:
  static void Transform(const InType* in, const Quantize& params,
                        OutType* output) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
    std::cout << "Quantize::Transform(" << std::string(typeid(InType).name())
              << ", " << std::string(typeid(OutType).name()) << ") -- "
              << kernel_size << "x" << leftovers << std::endl;
#endif
#else
    std::cerr << "FATAL: Quantize::Transform not implemented." << std::endl;
    std::exit(1);
#endif
  }
};

template <typename InType, typename OutType, int kernel_size, int leftovers>
class Transform1DKernel<InType, OutType, Dequantize, kernel_size, leftovers> {
 public:
  static void Transform(const InType* in, const Dequantize& params,
                        OutType* output) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
    std::cout << "Dequantize::Transform(" << std::string(typeid(InType).name())
              << ", " << std::string(typeid(OutType).name()) << ") -- "
              << kernel_size << "x" << leftovers << std::endl;
#endif
#else
    std::cerr << "FATAL: Dequantize::Transform not implemented." << std::endl;
    std::exit(1);
#endif
  }
};

template <typename InType, typename OutType, int kernel_size, int leftovers>
class Transform1DKernel<InType, OutType, Requantize, kernel_size, leftovers> {
 public:
  static void Transform(const InType* in, const Requantize& params,
                        OutType* output) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
    std::cout << "Requantize::Transform(" << std::string(typeid(InType).name())
              << ", " << std::string(typeid(OutType).name()) << ") -- "
              << kernel_size << "x" << leftovers << std::endl;
#endif
#else
    std::cerr << "FATAL: Requantize::Transform not implemented." << std::endl;
    std::exit(1);
#endif
  }
};

template <typename InType, typename OutType, int kernel_size, int leftovers,
          typename Type>
class Transform1DKernel<InType, OutType, MinMax<Type>, kernel_size, leftovers> {
 public:
  static void Transform(const InType* in, const MinMax<Type>& params,
                        OutType* output) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
    std::cout << "MinMax::Transform(" << std::string(typeid(InType).name())
              << ", " << std::string(typeid(OutType).name()) << ") -- "
              << kernel_size << "x" << leftovers << std::endl;
#endif
#else
    std::cerr << "FATAL: MinMax::Transform not implemented." << std::endl;
    std::exit(1);
#endif
  }
};

template <typename InType, typename OutType, int kernel_size, int leftovers,
          typename Type>
class Transform1DKernel<InType, OutType, BiasAdd<Type>, kernel_size,
                        leftovers> {
 public:
  static void Transform(const InType* in, const BiasAdd<Type>& params,
                        OutType* output) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
    std::cout << "BiasAdd::Transform(" << std::string(typeid(InType).name())
              << ", " << std::string(typeid(OutType).name()) << ") -- "
              << kernel_size << "x" << leftovers << std::endl;
#endif
#else
    std::cerr << "FATAL: BiasAdd::Transform not implemented." << std::endl;
    std::exit(1);
#endif
  }
};

template <typename InType, typename OutType>
class Transform1DUtil<InType, OutType, Quantize> {
 public:
  static int EstimateComputeCost(const Quantize& params) {
    return params.count * 8;
  }

  static const InType* OffsetInput(const Quantize& params, const InType* input,
                                   int offset) {
    return input + offset;
  }

  static OutType* OffsetOutput(const Quantize& params, OutType* output,
                               int offset) {
    return output + offset;
  }
};

template <typename InType, typename OutType>
class Transform1DUtil<InType, OutType, Requantize> {
 public:
  static int EstimateComputeCost(const Requantize& params) {
    return params.count * 12;
  }

  static const InType* OffsetInput(const Requantize& params,
                                   const InType* input, int offset) {
    return input + offset;
  }

  static OutType* OffsetOutput(const Requantize& params, OutType* output,
                               int offset) {
    return output + offset;
  }
};

template <typename InType, typename OutType>
class Transform1DUtil<InType, OutType, Dequantize> {
 public:
  static int EstimateComputeCost(const Dequantize& params) {
    return params.count * 12;
  }

  static const InType* OffsetInput(const Dequantize& params,
                                   const InType* input, int offset) {
    return input + offset;
  }

  static OutType* OffsetOutput(const Dequantize& params, OutType* output,
                               int offset) {
    return output + offset;
  }
};

template <typename InType, typename OutType, typename MinMaxType>
class Transform1DUtil<InType, OutType, MinMax<MinMaxType>> {
 public:
  static int EstimateComputeCost(const MinMax<MinMaxType>& params) {
    return params.count * 4;
  }

  static const InType* OffsetInput(const MinMax<MinMaxType>& params,
                                   const InType* input, int offset) {
    return input + offset;
  }

  static OutType* OffsetOutput(const MinMax<MinMaxType>& params,
                               OutType* output, int offset) {
    return output + offset;
  }
};

}  // namespace meta
}  // namespace gemmlowp

#ifdef GEMMLOWP_NEON_32
#include "transform_kernels_arm_32.h"
#elif defined(GEMMLOWP_NEON_64)
#include "transform_kernels_arm_64.h"
#endif

#endif  // GEMMLOWP_META_TRANSFORM_KERNELS_H_
