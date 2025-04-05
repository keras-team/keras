// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2021 The Eigen Team
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_SERIALIZER_H
#define EIGEN_SERIALIZER_H

#include <type_traits>

// The Serializer class encodes data into a memory buffer so it can be later
// reconstructed. This is mainly used to send objects back-and-forth between
// the CPU and GPU.

namespace Eigen {

/**
 * Serializes an object to a memory buffer.
 *
 * Useful for transferring data (e.g. back-and-forth to a device).
 */
template <typename T, typename EnableIf = void>
class Serializer;

// Specialization for POD types.
template <typename T>
class Serializer<T, typename std::enable_if_t<std::is_trivial<T>::value && std::is_standard_layout<T>::value>> {
 public:
  /**
   * Determines the required size of the serialization buffer for a value.
   *
   * \param value the value to serialize.
   * \return the required size.
   */
  EIGEN_DEVICE_FUNC size_t size(const T& value) const { return sizeof(value); }

  /**
   * Serializes a value to a byte buffer.
   * \param dest the destination buffer; if this is nullptr, does nothing.
   * \param end the end of the destination buffer.
   * \param value the value to serialize.
   * \return the next memory address past the end of the serialized data.
   */
  EIGEN_DEVICE_FUNC uint8_t* serialize(uint8_t* dest, uint8_t* end, const T& value) {
    if (EIGEN_PREDICT_FALSE(dest == nullptr)) return nullptr;
    if (EIGEN_PREDICT_FALSE(dest + sizeof(value) > end)) return nullptr;
    EIGEN_USING_STD(memcpy)
    memcpy(dest, &value, sizeof(value));
    return dest + sizeof(value);
  }

  /**
   * Deserializes a value from a byte buffer.
   * \param src the source buffer; if this is nullptr, does nothing.
   * \param end the end of the source buffer.
   * \param value the value to populate.
   * \return the next unprocessed memory address; nullptr if parsing errors are detected.
   */
  EIGEN_DEVICE_FUNC const uint8_t* deserialize(const uint8_t* src, const uint8_t* end, T& value) const {
    if (EIGEN_PREDICT_FALSE(src == nullptr)) return nullptr;
    if (EIGEN_PREDICT_FALSE(src + sizeof(value) > end)) return nullptr;
    EIGEN_USING_STD(memcpy)
    memcpy(&value, src, sizeof(value));
    return src + sizeof(value);
  }
};

// Specialization for DenseBase.
// Serializes [rows, cols, data...].
template <typename Derived>
class Serializer<DenseBase<Derived>, void> {
 public:
  typedef typename Derived::Scalar Scalar;

  struct Header {
    typename Derived::Index rows;
    typename Derived::Index cols;
  };

  EIGEN_DEVICE_FUNC size_t size(const Derived& value) const { return sizeof(Header) + sizeof(Scalar) * value.size(); }

  EIGEN_DEVICE_FUNC uint8_t* serialize(uint8_t* dest, uint8_t* end, const Derived& value) {
    if (EIGEN_PREDICT_FALSE(dest == nullptr)) return nullptr;
    if (EIGEN_PREDICT_FALSE(dest + size(value) > end)) return nullptr;
    const size_t header_bytes = sizeof(Header);
    const size_t data_bytes = sizeof(Scalar) * value.size();
    Header header = {value.rows(), value.cols()};
    EIGEN_USING_STD(memcpy)
    memcpy(dest, &header, header_bytes);
    dest += header_bytes;
    memcpy(dest, value.data(), data_bytes);
    return dest + data_bytes;
  }

  EIGEN_DEVICE_FUNC const uint8_t* deserialize(const uint8_t* src, const uint8_t* end, Derived& value) const {
    if (EIGEN_PREDICT_FALSE(src == nullptr)) return nullptr;
    if (EIGEN_PREDICT_FALSE(src + sizeof(Header) > end)) return nullptr;
    const size_t header_bytes = sizeof(Header);
    Header header;
    EIGEN_USING_STD(memcpy)
    memcpy(&header, src, header_bytes);
    src += header_bytes;
    const size_t data_bytes = sizeof(Scalar) * header.rows * header.cols;
    if (EIGEN_PREDICT_FALSE(src + data_bytes > end)) return nullptr;
    value.resize(header.rows, header.cols);
    memcpy(value.data(), src, data_bytes);
    return src + data_bytes;
  }
};

template <typename Scalar, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
class Serializer<Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>>
    : public Serializer<DenseBase<Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>>> {};

template <typename Scalar, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
class Serializer<Array<Scalar, Rows, Cols, Options, MaxRows, MaxCols>>
    : public Serializer<DenseBase<Array<Scalar, Rows, Cols, Options, MaxRows, MaxCols>>> {};

namespace internal {

// Recursive serialization implementation helper.
template <size_t N, typename... Types>
struct serialize_impl;

template <size_t N, typename T1, typename... Ts>
struct serialize_impl<N, T1, Ts...> {
  using Serializer = Eigen::Serializer<typename std::decay<T1>::type>;

  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE size_t serialize_size(const T1& value, const Ts&... args) {
    Serializer serializer;
    size_t size = serializer.size(value);
    return size + serialize_impl<N - 1, Ts...>::serialize_size(args...);
  }

  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE uint8_t* serialize(uint8_t* dest, uint8_t* end, const T1& value,
                                                                  const Ts&... args) {
    Serializer serializer;
    dest = serializer.serialize(dest, end, value);
    return serialize_impl<N - 1, Ts...>::serialize(dest, end, args...);
  }

  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const uint8_t* deserialize(const uint8_t* src, const uint8_t* end,
                                                                          T1& value, Ts&... args) {
    Serializer serializer;
    src = serializer.deserialize(src, end, value);
    return serialize_impl<N - 1, Ts...>::deserialize(src, end, args...);
  }
};

// Base case.
template <>
struct serialize_impl<0> {
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE size_t serialize_size() { return 0; }

  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE uint8_t* serialize(uint8_t* dest, uint8_t* /*end*/) { return dest; }

  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const uint8_t* deserialize(const uint8_t* src, const uint8_t* /*end*/) {
    return src;
  }
};

}  // namespace internal

/**
 * Determine the buffer size required to serialize a set of values.
 *
 * \param args ... arguments to serialize in sequence.
 * \return the total size of the required buffer.
 */
template <typename... Args>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE size_t serialize_size(const Args&... args) {
  return internal::serialize_impl<sizeof...(args), Args...>::serialize_size(args...);
}

/**
 * Serialize a set of values to the byte buffer.
 *
 * \param dest output byte buffer; if this is nullptr, does nothing.
 * \param end the end of the output byte buffer.
 * \param args ... arguments to serialize in sequence.
 * \return the next address after all serialized values.
 */
template <typename... Args>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE uint8_t* serialize(uint8_t* dest, uint8_t* end, const Args&... args) {
  return internal::serialize_impl<sizeof...(args), Args...>::serialize(dest, end, args...);
}

/**
 * Deserialize a set of values from the byte buffer.
 *
 * \param src input byte buffer; if this is nullptr, does nothing.
 * \param end the end of input byte buffer.
 * \param args ... arguments to deserialize in sequence.
 * \return the next address after all parsed values; nullptr if parsing errors are detected.
 */
template <typename... Args>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const uint8_t* deserialize(const uint8_t* src, const uint8_t* end,
                                                                 Args&... args) {
  return internal::serialize_impl<sizeof...(args), Args...>::deserialize(src, end, args...);
}

}  // namespace Eigen

#endif  // EIGEN_SERIALIZER_H
