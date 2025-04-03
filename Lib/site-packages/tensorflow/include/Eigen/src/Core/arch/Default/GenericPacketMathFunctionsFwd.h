// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2019 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_ARCH_GENERIC_PACKET_MATH_FUNCTIONS_FWD_H
#define EIGEN_ARCH_GENERIC_PACKET_MATH_FUNCTIONS_FWD_H

// IWYU pragma: private
#include "../../InternalHeaderCheck.h"

namespace Eigen {
namespace internal {

// Forward declarations of the generic math functions
// implemented in GenericPacketMathFunctions.h
// This is needed to workaround a circular dependency.

/***************************************************************************
 * Some generic implementations to be used by implementors
 ***************************************************************************/

/** Default implementation of pfrexp.
 * It is expected to be called by implementers of template<> pfrexp.
 */
template <typename Packet>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Packet pfrexp_generic(const Packet& a, Packet& exponent);

// Extracts the biased exponent value from Packet p, and casts the results to
// a floating-point Packet type. Used by pfrexp_generic. Override this if
// there is no unpacket_traits<Packet>::integer_packet.
template <typename Packet>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Packet pfrexp_generic_get_biased_exponent(const Packet& p);

/** Default implementation of pldexp.
 * It is expected to be called by implementers of template<> pldexp.
 */
template <typename Packet>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Packet pldexp_generic(const Packet& a, const Packet& exponent);

/** \internal \returns log(x) for single precision float */
template <typename Packet>
EIGEN_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS Packet plog_float(const Packet _x);

/** \internal \returns log2(x) for single precision float */
template <typename Packet>
EIGEN_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS Packet plog2_float(const Packet _x);

/** \internal \returns log(x) for single precision float */
template <typename Packet>
EIGEN_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS Packet plog_double(const Packet _x);

/** \internal \returns log2(x) for single precision float */
template <typename Packet>
EIGEN_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS Packet plog2_double(const Packet _x);

/** \internal \returns log(1 + x) */
template <typename Packet>
Packet generic_plog1p(const Packet& x);

/** \internal \returns exp(x)-1 */
template <typename Packet>
Packet generic_expm1(const Packet& x);

/** \internal \returns exp(x) for single precision float */
template <typename Packet>
EIGEN_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS Packet pexp_float(const Packet _x);

/** \internal \returns exp(x) for double precision real numbers */
template <typename Packet>
EIGEN_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS Packet pexp_double(const Packet _x);

/** \internal \returns sin(x) for single precision float */
template <typename Packet>
EIGEN_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS Packet psin_float(const Packet& x);

/** \internal \returns cos(x) for single precision float */
template <typename Packet>
EIGEN_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS Packet pcos_float(const Packet& x);

/** \internal \returns sin(x) for double precision float */
template <typename Packet>
EIGEN_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS Packet psin_double(const Packet& x);

/** \internal \returns cos(x) for double precision float */
template <typename Packet>
EIGEN_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS Packet pcos_double(const Packet& x);

/** \internal \returns asin(x) for single precision float */
template <typename Packet>
EIGEN_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS Packet pasin_float(const Packet& x);

/** \internal \returns acos(x) for single precision float */
template <typename Packet>
EIGEN_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS Packet pacos_float(const Packet& x);

/** \internal \returns atan(x) for single precision float */
template <typename Packet>
EIGEN_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS Packet patan_float(const Packet& x);

/** \internal \returns atan(x) for double precision float */
template <typename Packet>
EIGEN_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS Packet patan_double(const Packet& x);

/** \internal \returns tanh(x) for single precision float */
template <typename Packet>
EIGEN_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS Packet ptanh_float(const Packet& x);

/** \internal \returns atanh(x) for single precision float */
template <typename Packet>
EIGEN_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS Packet patanh_float(const Packet& x);

/** \internal \returns sqrt(x) for complex types */
template <typename Packet>
EIGEN_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS Packet psqrt_complex(const Packet& a);

/** \internal \returns x / y for complex types */
template <typename Packet>
EIGEN_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS Packet pdiv_complex(const Packet& x, const Packet& y);

template <typename Packet, int N>
struct ppolevl;

/** \internal \returns log(x) for complex types */
template <typename Packet>
EIGEN_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS Packet plog_complex(const Packet& x);

/** \internal \returns exp(x) for complex types */
template <typename Packet>
EIGEN_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS Packet pexp_complex(const Packet& x);

template <typename Packet>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet generic_rint(const Packet& a);

template <typename Packet>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet generic_floor(const Packet& a);

template <typename Packet>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet generic_ceil(const Packet& a);

template <typename Packet>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet generic_trunc(const Packet& a);

template <typename Packet>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet generic_round(const Packet& a);

// Macros for instantiating these generic functions for different backends.
#define EIGEN_PACKET_FUNCTION(METHOD, SCALAR, PACKET)                                             \
  template <>                                                                                     \
  EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC EIGEN_UNUSED PACKET p##METHOD<PACKET>(const PACKET& _x) { \
    return p##METHOD##_##SCALAR(_x);                                                              \
  }

#define EIGEN_FLOAT_PACKET_FUNCTION(METHOD, PACKET) EIGEN_PACKET_FUNCTION(METHOD, float, PACKET)
#define EIGEN_DOUBLE_PACKET_FUNCTION(METHOD, PACKET) EIGEN_PACKET_FUNCTION(METHOD, double, PACKET)

#define EIGEN_INSTANTIATE_GENERIC_MATH_FUNCS_FLOAT(PACKET)                                     \
  EIGEN_FLOAT_PACKET_FUNCTION(sin, PACKET)                                                     \
  EIGEN_FLOAT_PACKET_FUNCTION(cos, PACKET)                                                     \
  EIGEN_FLOAT_PACKET_FUNCTION(asin, PACKET)                                                    \
  EIGEN_FLOAT_PACKET_FUNCTION(acos, PACKET)                                                    \
  EIGEN_FLOAT_PACKET_FUNCTION(atan, PACKET)                                                    \
  EIGEN_FLOAT_PACKET_FUNCTION(tanh, PACKET)                                                    \
  EIGEN_FLOAT_PACKET_FUNCTION(atanh, PACKET)                                                   \
  EIGEN_FLOAT_PACKET_FUNCTION(log, PACKET)                                                     \
  EIGEN_FLOAT_PACKET_FUNCTION(log2, PACKET)                                                    \
  EIGEN_FLOAT_PACKET_FUNCTION(exp, PACKET)                                                     \
  template <>                                                                                  \
  EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC EIGEN_UNUSED PACKET pexpm1<PACKET>(const PACKET& _x) { \
    return internal::generic_expm1(_x);                                                        \
  }                                                                                            \
  template <>                                                                                  \
  EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC EIGEN_UNUSED PACKET plog1p<PACKET>(const PACKET& _x) { \
    return internal::generic_plog1p(_x);                                                       \
  }

#define EIGEN_INSTANTIATE_GENERIC_MATH_FUNCS_DOUBLE(PACKET) \
  EIGEN_DOUBLE_PACKET_FUNCTION(atan, PACKET)                \
  EIGEN_DOUBLE_PACKET_FUNCTION(log, PACKET)                 \
  EIGEN_DOUBLE_PACKET_FUNCTION(sin, PACKET)                 \
  EIGEN_DOUBLE_PACKET_FUNCTION(cos, PACKET)                 \
  EIGEN_DOUBLE_PACKET_FUNCTION(log2, PACKET)                \
  EIGEN_DOUBLE_PACKET_FUNCTION(exp, PACKET)

}  // end namespace internal
}  // end namespace Eigen

#endif  // EIGEN_ARCH_GENERIC_PACKET_MATH_FUNCTIONS_FWD_H
