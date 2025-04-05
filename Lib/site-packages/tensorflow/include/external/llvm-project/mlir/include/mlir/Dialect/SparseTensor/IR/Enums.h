//===- Enums.h - Enums for the SparseTensor dialect -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Typedefs and enums shared between MLIR code for manipulating the
// IR, and the lightweight runtime support library for sparse tensor
// manipulations.  That is, all the enums are used to define the API
// of the runtime library and hence are also needed when generating
// calls into the runtime library.  Moveover, the `LevelType` enum
// is also used as the internal IR encoding of dimension level types,
// to avoid code duplication (e.g., for the predicates).
//
// This file also defines x-macros <https://en.wikipedia.org/wiki/X_Macro>
// so that we can generate variations of the public functions for each
// supported primary- and/or overhead-type.
//
// Because this file defines a library which is a dependency of the
// runtime library itself, this file must not depend on any MLIR internals
// (e.g., operators, attributes, ArrayRefs, etc) lest the runtime library
// inherit those dependencies.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SPARSETENSOR_IR_ENUMS_H
#define MLIR_DIALECT_SPARSETENSOR_IR_ENUMS_H

// NOTE: Client code will need to include "mlir/ExecutionEngine/Float16bits.h"
// if they want to use the `MLIR_SPARSETENSOR_FOREVERY_V` macro.

#include <cassert>
#include <cinttypes>
#include <complex>
#include <optional>
#include <vector>

namespace mlir {
namespace sparse_tensor {

/// This type is used in the public API at all places where MLIR expects
/// values with the built-in type "index".  For now, we simply assume that
/// type is 64-bit, but targets with different "index" bitwidths should
/// link with an alternatively built runtime support library.
using index_type = uint64_t;

/// Encoding of overhead types (both position overhead and coordinate
/// overhead), for "overloading" @newSparseTensor.
enum class OverheadType : uint32_t {
  kIndex = 0,
  kU64 = 1,
  kU32 = 2,
  kU16 = 3,
  kU8 = 4
};

// This x-macro calls its argument on every overhead type which has
// fixed-width.  It excludes `index_type` because that type is often
// handled specially (e.g., by translating it into the architecture-dependent
// equivalent fixed-width overhead type).
#define MLIR_SPARSETENSOR_FOREVERY_FIXED_O(DO)                                 \
  DO(64, uint64_t)                                                             \
  DO(32, uint32_t)                                                             \
  DO(16, uint16_t)                                                             \
  DO(8, uint8_t)

// This x-macro calls its argument on every overhead type, including
// `index_type`.
#define MLIR_SPARSETENSOR_FOREVERY_O(DO)                                       \
  MLIR_SPARSETENSOR_FOREVERY_FIXED_O(DO)                                       \
  DO(0, index_type)

// These are not just shorthands but indicate the particular
// implementation used (e.g., as opposed to C99's `complex double`,
// or MLIR's `ComplexType`).
using complex64 = std::complex<double>;
using complex32 = std::complex<float>;

/// Encoding of the elemental type, for "overloading" @newSparseTensor.
enum class PrimaryType : uint32_t {
  kF64 = 1,
  kF32 = 2,
  kF16 = 3,
  kBF16 = 4,
  kI64 = 5,
  kI32 = 6,
  kI16 = 7,
  kI8 = 8,
  kC64 = 9,
  kC32 = 10
};

// This x-macro includes all `V` types.
#define MLIR_SPARSETENSOR_FOREVERY_V(DO)                                       \
  DO(F64, double)                                                              \
  DO(F32, float)                                                               \
  DO(F16, f16)                                                                 \
  DO(BF16, bf16)                                                               \
  DO(I64, int64_t)                                                             \
  DO(I32, int32_t)                                                             \
  DO(I16, int16_t)                                                             \
  DO(I8, int8_t)                                                               \
  DO(C64, complex64)                                                           \
  DO(C32, complex32)

// This x-macro includes all `V` types and supports variadic arguments.
#define MLIR_SPARSETENSOR_FOREVERY_V_VAR(DO, ...)                              \
  DO(F64, double, __VA_ARGS__)                                                 \
  DO(F32, float, __VA_ARGS__)                                                  \
  DO(F16, f16, __VA_ARGS__)                                                    \
  DO(BF16, bf16, __VA_ARGS__)                                                  \
  DO(I64, int64_t, __VA_ARGS__)                                                \
  DO(I32, int32_t, __VA_ARGS__)                                                \
  DO(I16, int16_t, __VA_ARGS__)                                                \
  DO(I8, int8_t, __VA_ARGS__)                                                  \
  DO(C64, complex64, __VA_ARGS__)                                              \
  DO(C32, complex32, __VA_ARGS__)

// This x-macro calls its argument on every pair of overhead and `V` types.
#define MLIR_SPARSETENSOR_FOREVERY_V_O(DO)                                     \
  MLIR_SPARSETENSOR_FOREVERY_V_VAR(DO, 64, uint64_t)                           \
  MLIR_SPARSETENSOR_FOREVERY_V_VAR(DO, 32, uint32_t)                           \
  MLIR_SPARSETENSOR_FOREVERY_V_VAR(DO, 16, uint16_t)                           \
  MLIR_SPARSETENSOR_FOREVERY_V_VAR(DO, 8, uint8_t)                             \
  MLIR_SPARSETENSOR_FOREVERY_V_VAR(DO, 0, index_type)

constexpr bool isFloatingPrimaryType(PrimaryType valTy) {
  return PrimaryType::kF64 <= valTy && valTy <= PrimaryType::kBF16;
}

constexpr bool isIntegralPrimaryType(PrimaryType valTy) {
  return PrimaryType::kI64 <= valTy && valTy <= PrimaryType::kI8;
}

constexpr bool isRealPrimaryType(PrimaryType valTy) {
  return PrimaryType::kF64 <= valTy && valTy <= PrimaryType::kI8;
}

constexpr bool isComplexPrimaryType(PrimaryType valTy) {
  return PrimaryType::kC64 <= valTy && valTy <= PrimaryType::kC32;
}

/// The actions performed by @newSparseTensor.
enum class Action : uint32_t {
  kEmpty = 0,
  kFromReader = 1,
  kPack = 2,
  kSortCOOInPlace = 3,
};

/// This enum defines all supported storage format without the level properties.
enum class LevelFormat : uint64_t {
  Undef = 0x00000000,
  Dense = 0x00010000,
  Batch = 0x00020000,
  Compressed = 0x00040000,
  Singleton = 0x00080000,
  LooseCompressed = 0x00100000,
  NOutOfM = 0x00200000,
};

constexpr bool encPowOfTwo(LevelFormat fmt) {
  auto enc = static_cast<std::underlying_type_t<LevelFormat>>(fmt);
  return (enc & (enc - 1)) == 0;
}

// All LevelFormats must have only one bit set (power of two).
static_assert(encPowOfTwo(LevelFormat::Dense) &&
              encPowOfTwo(LevelFormat::Batch) &&
              encPowOfTwo(LevelFormat::Compressed) &&
              encPowOfTwo(LevelFormat::Singleton) &&
              encPowOfTwo(LevelFormat::LooseCompressed) &&
              encPowOfTwo(LevelFormat::NOutOfM));

template <LevelFormat... targets>
constexpr bool isAnyOfFmt(LevelFormat fmt) {
  return (... || (targets == fmt));
}

/// Returns string representation of the given level format.
constexpr const char *toFormatString(LevelFormat lvlFmt) {
  switch (lvlFmt) {
  case LevelFormat::Undef:
    return "undef";
  case LevelFormat::Dense:
    return "dense";
  case LevelFormat::Batch:
    return "batch";
  case LevelFormat::Compressed:
    return "compressed";
  case LevelFormat::Singleton:
    return "singleton";
  case LevelFormat::LooseCompressed:
    return "loose_compressed";
  case LevelFormat::NOutOfM:
    return "structured";
  }
  return "";
}

/// This enum defines all the nondefault properties for storage formats.
enum class LevelPropNonDefault : uint64_t {
  Nonunique = 0x0001,  // 0b001
  Nonordered = 0x0002, // 0b010
  SoA = 0x0004,        // 0b100
};

/// Returns string representation of the given level properties.
constexpr const char *toPropString(LevelPropNonDefault lvlProp) {
  switch (lvlProp) {
  case LevelPropNonDefault::Nonunique:
    return "nonunique";
  case LevelPropNonDefault::Nonordered:
    return "nonordered";
  case LevelPropNonDefault::SoA:
    return "soa";
  }
  return "";
}

/// This enum defines all the sparse representations supportable by
/// the SparseTensor dialect. We use a lightweight encoding to encode
/// the "format" per se (dense, compressed, singleton, loose_compressed,
/// n-out-of-m), the "properties" (ordered, unique) as well as n and m when
/// the format is NOutOfM.
/// The encoding is chosen for performance of the runtime library, and thus may
/// change in future versions; consequently, client code should use the
/// predicate functions defined below, rather than relying on knowledge
/// about the particular binary encoding.
///
/// The `Undef` "format" is a special value used internally for cases
/// where we need to store an undefined or indeterminate `LevelType`.
/// It should not be used externally, since it does not indicate an
/// actual/representable format.

struct LevelType {
public:
  /// Check that the `LevelType` contains a valid (possibly undefined) value.
  static constexpr bool isValidLvlBits(uint64_t lvlBits) {
    auto fmt = static_cast<LevelFormat>(lvlBits & 0xffff0000);
    const uint64_t propertyBits = lvlBits & 0xffff;
    // If undefined/dense/batch/NOutOfM, then must be unique and ordered.
    // Otherwise, the format must be one of the known ones.
    return (isAnyOfFmt<LevelFormat::Undef, LevelFormat::Dense,
                       LevelFormat::Batch, LevelFormat::NOutOfM>(fmt))
               ? (propertyBits == 0)
               : (isAnyOfFmt<LevelFormat::Compressed, LevelFormat::Singleton,
                             LevelFormat::LooseCompressed>(fmt));
  }

  /// Convert a LevelFormat to its corresponding LevelType with the given
  /// properties. Returns std::nullopt when the properties are not applicable
  /// for the input level format.
  static std::optional<LevelType>
  buildLvlType(LevelFormat lf,
               const std::vector<LevelPropNonDefault> &properties,
               uint64_t n = 0, uint64_t m = 0) {
    assert((n & 0xff) == n && (m & 0xff) == m);
    uint64_t newN = n << 32;
    uint64_t newM = m << 40;
    uint64_t ltBits = static_cast<uint64_t>(lf) | newN | newM;
    for (auto p : properties)
      ltBits |= static_cast<uint64_t>(p);

    return isValidLvlBits(ltBits) ? std::optional(LevelType(ltBits))
                                  : std::nullopt;
  }
  static std::optional<LevelType> buildLvlType(LevelFormat lf, bool ordered,
                                               bool unique, uint64_t n = 0,
                                               uint64_t m = 0) {
    std::vector<LevelPropNonDefault> properties;
    if (!ordered)
      properties.push_back(LevelPropNonDefault::Nonordered);
    if (!unique)
      properties.push_back(LevelPropNonDefault::Nonunique);
    return buildLvlType(lf, properties, n, m);
  }

  /// Explicit conversion from uint64_t.
  constexpr explicit LevelType(uint64_t bits) : lvlBits(bits) {
    assert(isValidLvlBits(bits));
  };

  /// Constructs a LevelType with the given format using all default properties.
  /*implicit*/ LevelType(LevelFormat f) : lvlBits(static_cast<uint64_t>(f)) {
    assert(isValidLvlBits(lvlBits) && !isa<LevelFormat::NOutOfM>());
  };

  /// Converts to uint64_t
  explicit operator uint64_t() const { return lvlBits; }

  bool operator==(const LevelType lhs) const {
    return static_cast<uint64_t>(lhs) == lvlBits;
  }
  bool operator!=(const LevelType lhs) const { return !(*this == lhs); }

  LevelType stripStorageIrrelevantProperties() const {
    // Properties other than `SoA` do not change the storage scheme of the
    // sparse tensor.
    constexpr uint64_t mask =
        0xffff & ~static_cast<uint64_t>(LevelPropNonDefault::SoA);
    return LevelType(lvlBits & ~mask);
  }

  /// Get N of NOutOfM level type.
  constexpr uint64_t getN() const {
    assert(isa<LevelFormat::NOutOfM>());
    return (lvlBits >> 32) & 0xff;
  }

  /// Get M of NOutOfM level type.
  constexpr uint64_t getM() const {
    assert(isa<LevelFormat::NOutOfM>());
    return (lvlBits >> 40) & 0xff;
  }

  /// Get the `LevelFormat` of the `LevelType`.
  constexpr LevelFormat getLvlFmt() const {
    return static_cast<LevelFormat>(lvlBits & 0xffff0000);
  }

  /// Check if the `LevelType` is in the `LevelFormat`.
  template <LevelFormat... fmt>
  constexpr bool isa() const {
    return (... || (getLvlFmt() == fmt)) || false;
  }

  /// Check if the `LevelType` has the properties
  template <LevelPropNonDefault p>
  constexpr bool isa() const {
    return lvlBits & static_cast<uint64_t>(p);
  }

  /// Check if the `LevelType` is considered to be sparse.
  constexpr bool hasSparseSemantic() const {
    return isa<LevelFormat::Compressed, LevelFormat::Singleton,
               LevelFormat::LooseCompressed, LevelFormat::NOutOfM>();
  }

  /// Check if the `LevelType` is considered to be dense-like.
  constexpr bool hasDenseSemantic() const {
    return isa<LevelFormat::Dense, LevelFormat::Batch>();
  }

  /// Check if the `LevelType` needs positions array.
  constexpr bool isWithPosLT() const {
    assert(!isa<LevelFormat::Undef>());
    return isa<LevelFormat::Compressed, LevelFormat::LooseCompressed>();
  }

  /// Check if the `LevelType` needs coordinates array.
  constexpr bool isWithCrdLT() const {
    assert(!isa<LevelFormat::Undef>());
    // All sparse levels has coordinate array.
    return hasSparseSemantic();
  }

  constexpr unsigned getNumBuffer() const {
    return hasDenseSemantic() ? 0 : (isWithPosLT() ? 2 : 1);
  }

  std::string toMLIRString() const {
    std::string lvlStr = toFormatString(getLvlFmt());
    std::string propStr = "";
    if (isa<LevelFormat::NOutOfM>()) {
      lvlStr +=
          "[" + std::to_string(getN()) + ", " + std::to_string(getM()) + "]";
    }
    if (isa<LevelPropNonDefault::Nonunique>())
      propStr += toPropString(LevelPropNonDefault::Nonunique);

    if (isa<LevelPropNonDefault::Nonordered>()) {
      if (!propStr.empty())
        propStr += ", ";
      propStr += toPropString(LevelPropNonDefault::Nonordered);
    }
    if (isa<LevelPropNonDefault::SoA>()) {
      if (!propStr.empty())
        propStr += ", ";
      propStr += toPropString(LevelPropNonDefault::SoA);
    }
    if (!propStr.empty())
      lvlStr += ("(" + propStr + ")");
    return lvlStr;
  }

private:
  /// Bit manipulations for LevelType:
  ///
  /// | 8-bit n | 8-bit m | 16-bit LevelFormat | 16-bit LevelProperty |
  ///
  uint64_t lvlBits;
};

// For backward-compatibility. TODO: remove below after fully migration.
constexpr uint64_t nToBits(uint64_t n) { return n << 32; }
constexpr uint64_t mToBits(uint64_t m) { return m << 40; }

inline std::optional<LevelType>
buildLevelType(LevelFormat lf,
               const std::vector<LevelPropNonDefault> &properties,
               uint64_t n = 0, uint64_t m = 0) {
  return LevelType::buildLvlType(lf, properties, n, m);
}
inline std::optional<LevelType> buildLevelType(LevelFormat lf, bool ordered,
                                               bool unique, uint64_t n = 0,
                                               uint64_t m = 0) {
  return LevelType::buildLvlType(lf, ordered, unique, n, m);
}
inline bool isUndefLT(LevelType lt) { return lt.isa<LevelFormat::Undef>(); }
inline bool isDenseLT(LevelType lt) { return lt.isa<LevelFormat::Dense>(); }
inline bool isBatchLT(LevelType lt) { return lt.isa<LevelFormat::Batch>(); }
inline bool isCompressedLT(LevelType lt) {
  return lt.isa<LevelFormat::Compressed>();
}
inline bool isLooseCompressedLT(LevelType lt) {
  return lt.isa<LevelFormat::LooseCompressed>();
}
inline bool isSingletonLT(LevelType lt) {
  return lt.isa<LevelFormat::Singleton>();
}
inline bool isNOutOfMLT(LevelType lt) { return lt.isa<LevelFormat::NOutOfM>(); }
inline bool isOrderedLT(LevelType lt) {
  return !lt.isa<LevelPropNonDefault::Nonordered>();
}
inline bool isUniqueLT(LevelType lt) {
  return !lt.isa<LevelPropNonDefault::Nonunique>();
}
inline bool isWithCrdLT(LevelType lt) { return lt.isWithCrdLT(); }
inline bool isWithPosLT(LevelType lt) { return lt.isWithPosLT(); }
inline bool isValidLT(LevelType lt) {
  return LevelType::isValidLvlBits(static_cast<uint64_t>(lt));
}
inline std::optional<LevelFormat> getLevelFormat(LevelType lt) {
  LevelFormat fmt = lt.getLvlFmt();
  if (fmt == LevelFormat::Undef)
    return std::nullopt;
  return fmt;
}
inline uint64_t getN(LevelType lt) { return lt.getN(); }
inline uint64_t getM(LevelType lt) { return lt.getM(); }
inline bool isValidNOutOfMLT(LevelType lt, uint64_t n, uint64_t m) {
  return isNOutOfMLT(lt) && lt.getN() == n && lt.getM() == m;
}
inline std::string toMLIRString(LevelType lt) { return lt.toMLIRString(); }

/// Bit manipulations for affine encoding.
///
/// Note that because the indices in the mappings refer to dimensions
/// and levels (and *not* the sizes of these dimensions and levels), the
/// 64-bit encoding gives ample room for a compact encoding of affine
/// operations in the higher bits. Pure permutations still allow for
/// 60-bit indices. But non-permutations reserve 20-bits for the
/// potential three components (index i, constant, index ii).
///
/// The compact encoding is as follows:
///
///  0xffffffffffffffff
/// |0000      |                        60-bit idx| e.g. i
/// |0001 floor|           20-bit const|20-bit idx| e.g. i floor c
/// |0010 mod  |           20-bit const|20-bit idx| e.g. i mod c
/// |0011 mul  |20-bit idx|20-bit const|20-bit idx| e.g. i + c * ii
///
/// This encoding provides sufficient generality for currently supported
/// sparse tensor types. To generalize this more, we will need to provide
/// a broader encoding scheme for affine functions. Also, the library
/// encoding may be replaced with pure "direct-IR" code in the future.
///
constexpr uint64_t encodeDim(uint64_t i, uint64_t cf, uint64_t cm) {
  if (cf != 0) {
    assert(cf <= 0xfffffu && cm == 0 && i <= 0xfffffu);
    return (static_cast<uint64_t>(0x01u) << 60) | (cf << 20) | i;
  }
  if (cm != 0) {
    assert(cm <= 0xfffffu && i <= 0xfffffu);
    return (static_cast<uint64_t>(0x02u) << 60) | (cm << 20) | i;
  }
  assert(i <= 0x0fffffffffffffffu);
  return i;
}
constexpr uint64_t encodeLvl(uint64_t i, uint64_t c, uint64_t ii) {
  if (c != 0) {
    assert(c <= 0xfffffu && ii <= 0xfffffu && i <= 0xfffffu);
    return (static_cast<uint64_t>(0x03u) << 60) | (c << 20) | (ii << 40) | i;
  }
  assert(i <= 0x0fffffffffffffffu);
  return i;
}
constexpr bool isEncodedFloor(uint64_t v) { return (v >> 60) == 0x01u; }
constexpr bool isEncodedMod(uint64_t v) { return (v >> 60) == 0x02u; }
constexpr bool isEncodedMul(uint64_t v) { return (v >> 60) == 0x03u; }
constexpr uint64_t decodeIndex(uint64_t v) { return v & 0xfffffu; }
constexpr uint64_t decodeConst(uint64_t v) { return (v >> 20) & 0xfffffu; }
constexpr uint64_t decodeMulc(uint64_t v) { return (v >> 20) & 0xfffffu; }
constexpr uint64_t decodeMuli(uint64_t v) { return (v >> 40) & 0xfffffu; }

} // namespace sparse_tensor
} // namespace mlir

#endif // MLIR_DIALECT_SPARSETENSOR_IR_ENUMS_H
