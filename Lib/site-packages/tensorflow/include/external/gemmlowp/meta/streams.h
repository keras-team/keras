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

#ifndef GEMMLOWP_META_STREAMS_H_
#define GEMMLOWP_META_STREAMS_H_

#include <iostream>
#include <typeinfo>
#include "base.h"

namespace gemmlowp {
namespace meta {

struct RowMajor {
 public:
  int count;
  int stride;
};

struct RowMajorWithSum {
 public:
  int count;
  int stride;
  int multiplicative_sum_offset;
  int additive_sum_offset;
};

struct ColumnMajorWithSum {
 public:
  int count;
  int stride;
  int multiplicative_sum_offset;
  int additive_sum_offset;
};

template <typename InType>
class StreamUtil<InType, RowMajor> {
 public:
  static const InType* Offset(const RowMajor& params, const InType* source,
                              int offset_stride, int offset_advance) {
    return reinterpret_cast<const InType*>(
        reinterpret_cast<const std::uint8_t*>(source) +
        offset_stride * params.stride + offset_advance * sizeof(InType));
  }

  static InType* Offset(const RowMajor& params, InType* source,
                        int offset_stride, int offset_advance) {
    return reinterpret_cast<InType*>(reinterpret_cast<std::uint8_t*>(source) +
                                     offset_stride * params.stride +
                                     offset_advance * sizeof(InType));
  }

  static int Scratch(const RowMajor& params, int lanes_count, int pack_size) {
    return AlignTo<64>(lanes_count * AlignTo(pack_size, params.stride));
  }
};

template <typename InType>
class StreamUtil<InType, RowMajorWithSum> {
 public:
  static const InType* Offset(const RowMajorWithSum& params,
                              const InType* source, int offset_stride,
                              int offset_advance) {
    return reinterpret_cast<const InType*>(
        reinterpret_cast<const std::uint8_t*>(source) +
        offset_stride * params.stride + offset_advance * sizeof(InType));
  }

  static InType* Offset(const RowMajorWithSum& params, InType* source,
                        int offset_stride, int offset_advance) {
    return reinterpret_cast<InType*>(reinterpret_cast<std::uint8_t*>(source) +
                                     offset_stride * params.stride +
                                     offset_advance * sizeof(InType));
  }

  static int Scratch(const RowMajorWithSum& params, int lanes_count,
                     int pack_size) {
    return 32 + AlignTo<32>(sizeof(InType) * lanes_count *
                            AlignTo(pack_size, params.count));
  }
};

template <typename InType>
class StreamUtil<InType, ColumnMajorWithSum> {
 public:
  static const InType* Offset(const ColumnMajorWithSum& params,
                              const InType* source, int offset_stride,
                              int offset_advance) {
    return reinterpret_cast<const InType*>(
        reinterpret_cast<const std::uint8_t*>(source) +
        params.stride * offset_advance + offset_stride * sizeof(InType));
  }

  static const InType* Offset(const ColumnMajorWithSum& params, InType* source,
                              int offset_stride, int offset_advance) {
    return reinterpret_cast<InType*>(reinterpret_cast<std::uint8_t*>(source) +
                                     params.stride * offset_advance +
                                     offset_stride * sizeof(InType));
  }

  static int Scratch(const ColumnMajorWithSum& params, int lanes_count,
                     int pack_size) {
    return 32 + AlignTo<32>(sizeof(InType) * lanes_count *
                            AlignTo(pack_size, params.count));
  }
};

template <typename InType, int lanes_count, int pack_size, int leftovers>
class Stream<InType, lanes_count, pack_size, leftovers, RowMajor> {
 public:
  static void Pack(const InType* in, const RowMajor& params, InType* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
    std::cout << "RowMajor(" << std::string(typeid(InType).name())
              << ")::Pack() -- " << lanes_count << "x" << pack_size << " + "
              << leftovers << std::endl;
#endif
#else
    if (lanes_count != 0) {
      std::cerr << "FATAL: RowMajorWithSum::Pack not implemented." << std::endl;
      std::exit(1);
    }
#endif
  }

  static int UnpackedAdvance(const RowMajor& params) {
    return sizeof(InType) * pack_size;
  }

  static int PackedAdvance(const RowMajor& params) {
    return sizeof(InType) * pack_size * lanes_count;
  }

  static int UnpackedStride(const RowMajor& params) {
    return lanes_count * params.stride;
  }

  static int PackedStride(const RowMajor& params) {
    return AlignTo<32>(lanes_count * AlignTo<pack_size>(params.stride));
  }

  static int Scratch(const RowMajor& params) { return PackedStride(params); }

#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  static void Debug(const RowMajor& params) {
    std::cout << "RowMajor(" << typeid(InType).name() << ")" << std::endl;
    std::cout << "  dims: " << lanes_count << "x" << pack_size << " + "
              << leftovers << std::endl;
    std::cout << "  scratch: " << Scratch(params) << std::endl;
    std::cout << "  unpacked advance: " << UnpackedAdvance(params) << std::endl;
    std::cout << "  packed advance: " << PackedAdvance(params) << std::endl;
    std::cout << "  unpacked stride: " << UnpackedStride(params) << std::endl;
    std::cout << "  packed stride: " << PackedStride(params) << std::endl;
    std::cout << "  params:" << std::endl;
    std::cout << "    count: " << params.count << std::endl;
    std::cout << "    stride: " << params.stride << std::endl;
  }
#endif
#endif
};

template <typename InType, int lanes_count, int pack_size, int leftovers>
class Stream<InType, lanes_count, pack_size, leftovers, RowMajorWithSum> {
 public:
  static void Pack(const InType* in, const RowMajorWithSum& params,
                   InType* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
    std::cout << "RowMajorWithSum(" << typeid(InType).name() << ")::Pack() -- "
              << lanes_count << "x" << pack_size << " + " << leftovers
              << std::endl;
#endif
#else
    if (lanes_count != 0) {
      std::cerr << "FATAL: RowMajorWithSum::Pack not implemented." << std::endl;
      std::exit(1);
    }
#endif
  }

  static int UnpackedAdvance(const RowMajorWithSum& params) {
    return sizeof(InType) * pack_size;
  }

  static int PackedAdvance(const RowMajorWithSum& params) {
    return sizeof(InType) * pack_size * lanes_count;
  }

  static int UnpackedStride(const RowMajorWithSum& params) {
    return sizeof(InType) * lanes_count * params.stride;
  }

  static int PackedStride(const RowMajorWithSum& params) {
    return 32 + AlignTo<32>(sizeof(InType) * lanes_count *
                            AlignTo<pack_size>(params.count));
  }

  static int Scratch(const RowMajorWithSum& params) {
    return PackedStride(params);
  }

#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  static void Debug(const RowMajorWithSum& params) {
    std::cout << "RowMajorWithSum(" << typeid(InType).name() << ")"
              << std::endl;
    std::cout << "  dims: " << lanes_count << "x" << pack_size << " + "
              << leftovers << std::endl;
    std::cout << "  scratch: " << Scratch(params) << std::endl;
    std::cout << "  unpacked advance: " << UnpackedAdvance(params) << std::endl;
    std::cout << "  packed advance: " << PackedAdvance(params) << std::endl;
    std::cout << "  unpacked stride: " << UnpackedStride(params) << std::endl;
    std::cout << "  packed stride: " << PackedStride(params) << std::endl;
    std::cout << "  params:" << std::endl;
    std::cout << "    count: " << params.count << std::endl;
    std::cout << "    stride: " << params.stride << std::endl;
    std::cout << "    multiplicative_sum_offset: "
              << params.multiplicative_sum_offset << std::endl;
    std::cout << "    additive_sum_offset: " << params.additive_sum_offset
              << std::endl;
  }
#endif
#endif
};

template <typename InType, int lanes_count, int pack_size, int leftovers>
class Stream<InType, lanes_count, pack_size, leftovers, ColumnMajorWithSum> {
 public:
  static void Pack(const InType* in, const ColumnMajorWithSum& params,
                   InType* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
    std::cout << "ColumnMajorWithSum(" << typeid(InType).name()
              << ")::Pack() -- " << lanes_count << "x" << pack_size << " + "
              << leftovers << std::endl;
#endif
#else
    if (lanes_count != 0) {
      std::cerr << "FATAL: ColumnMajorWithSum::Pack not implemented."
                << std::endl;
      std::exit(1);
    }
#endif
  }

  static int UnpackedAdvance(const ColumnMajorWithSum& params) {
    return sizeof(InType) * pack_size * params.stride;
  }

  static int PackedAdvance(const ColumnMajorWithSum& params) {
    return sizeof(InType) * pack_size * lanes_count;
  }

  static int UnpackedStride(const ColumnMajorWithSum& params) {
    return sizeof(InType) * lanes_count;
  }

  static int PackedStride(const ColumnMajorWithSum& params) {
    return 32 + AlignTo<32>(sizeof(InType) * lanes_count *
                            AlignTo<pack_size>(params.count));
  }

  static int Scratch(const ColumnMajorWithSum& params) {
    return PackedStride(params);
  }

#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  static void Debug(const ColumnMajorWithSum& params) {
    std::cout << "ColumnMajorWithSum(" << typeid(InType).name() << ")"
              << std::endl;
    std::cout << "  dims: " << lanes_count << "x" << pack_size << " + "
              << leftovers << std::endl;
    std::cout << "  scratch: " << Scratch(params) << std::endl;
    std::cout << "  unpacked advance: " << UnpackedAdvance(params) << std::endl;
    std::cout << "  packed advance: " << PackedAdvance(params) << std::endl;
    std::cout << "  unpacked stride: " << UnpackedStride(params) << std::endl;
    std::cout << "  packed stride: " << PackedStride(params) << std::endl;
    std::cout << "  params:" << std::endl;
    std::cout << "    count: " << params.count << std::endl;
    std::cout << "    stride: " << params.stride << std::endl;
    std::cout << "    multiplicative_sum_offset: "
              << params.multiplicative_sum_offset << std::endl;
    std::cout << "    additive_sum_offset: " << params.additive_sum_offset
              << std::endl;
  }
#endif
#endif
};

}  // namespace meta
}  // namespace gemmlowp

#ifdef GEMMLOWP_NEON_32
#include "streams_arm_32.h"
#elif defined(GEMMLOWP_NEON_64)
#include "streams_arm_64.h"
#endif

#endif  // GEMMLOWP_META_STREAMS_H_
