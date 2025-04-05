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

#ifndef GEMMLOWP_META_BASE_H_
#define GEMMLOWP_META_BASE_H_

#include <cassert>
#include <cstdint>

#include "../internal/common.h"

namespace gemmlowp {
namespace meta {

template <int align>
inline int AlignTo(int value) {
  return ((value + align - 1) / align) * align;
}

inline int AlignTo(int align, int value) {
  return ((value + align - 1) / align) * align;
}

template <typename Kernel_, typename OutputStream_>
struct FusedKernelParams {
 public:
  typedef Kernel_ Kernel;
  typedef OutputStream_ OutputStream;

  Kernel kernel;
  OutputStream output_stream;
};

template <typename InType_, typename OutType_, typename LeftStream_,
          typename RightStream_, typename Kernel_, typename OutputStream_>
struct GemmParams {
 public:
  typedef InType_ InType;
  typedef OutType_ OutType;
  typedef LeftStream_ LeftStream;
  typedef RightStream_ RightStream;
  typedef Kernel_ Kernel;
  typedef OutputStream_ OutputStream;

  typedef FusedKernelParams<Kernel, OutputStream> FusedKernel;

  // Common parameters.

  int m;
  int n;
  int k;

  const InType* lhs;
  const InType* rhs;
  OutType* result;
  std::uint8_t* scratch;

  // Specialized parameters.

  LeftStream left_stream;
  RightStream right_stream;
  FusedKernel fused_kernel;
};

template <typename InType, int lanes_count, int pack_size, int leftovers,
          typename StreamParams>
class Stream {
 public:
  static void Pack(const InType* in, const StreamParams& params, InType* out);

  static int UnpackedAdvance(const StreamParams& params);

  static int PackedAdvance(const StreamParams& params);

  static int UnpackedStride(const StreamParams& params);

  static int PackedStride(const StreamParams& params);
};

template <typename InType, typename StreamType>
class StreamUtil {
 public:
  static const InType* Offset(const StreamType& params, const InType* source,
                              int offset_stride, int offset_advance);

  static int Scratch(const StreamType& params, int lanes);
};

template <typename InType, typename OutType, typename Kernel,
          typename OutputStream, int kernel_m, int kernel_n, int pack_size>
class MulKernel {
 public:
  static void Multiply(const InType* lhs, const InType* rhs,
                       const FusedKernelParams<Kernel, OutputStream>& params,
                       OutType* result);
};

template <typename InType_, typename OutType_, typename Kernel_>
struct Transform1DParams {
  typedef InType_ InType;
  typedef OutType_ OutType;
  typedef Kernel_ Kernel;

  const InType* input;
  OutType* output;
  std::uint8_t* scratch;

  Kernel kernel;
};

template <typename InType, typename OutType, typename Kernel, int kernel_size,
          int leftovers>
class Transform1DKernel {
 public:
  static void Transform(const InType* input, const Kernel& params,
                        OutType* output);
};

template <typename InType, typename OutType, typename Transform>
class Transform1DUtil {
 public:
  static int EstimateComputeCost(const Transform& params);

  static const InType* OffsetInput(const Transform& params, const InType* input,
                                   int offset);

  static OutType* OffsetOutput(const Transform& params, OutType* output,
                               int offset);
};

}  // namespace meta
}  // namespace gemmlowp

#endif  // GEMMLOWP_META_BASE_H_
