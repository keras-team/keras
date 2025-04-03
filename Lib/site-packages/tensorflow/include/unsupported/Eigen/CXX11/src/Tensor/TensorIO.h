// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_CXX11_TENSOR_TENSOR_IO_H
#define EIGEN_CXX11_TENSOR_TENSOR_IO_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {

struct TensorIOFormat;

namespace internal {
template <typename Tensor, std::size_t rank, typename Format, typename EnableIf = void>
struct TensorPrinter;
}

template <typename Derived_>
struct TensorIOFormatBase {
  using Derived = Derived_;
  TensorIOFormatBase(const std::vector<std::string>& separator, const std::vector<std::string>& prefix,
                     const std::vector<std::string>& suffix, int precision = StreamPrecision, int flags = 0,
                     const std::string& tenPrefix = "", const std::string& tenSuffix = "", const char fill = ' ')
      : tenPrefix(tenPrefix),
        tenSuffix(tenSuffix),
        prefix(prefix),
        suffix(suffix),
        separator(separator),
        fill(fill),
        precision(precision),
        flags(flags) {
    init_spacer();
  }

  void init_spacer() {
    if ((flags & DontAlignCols)) return;
    spacer.resize(prefix.size());
    spacer[0] = "";
    int i = int(tenPrefix.length()) - 1;
    while (i >= 0 && tenPrefix[i] != '\n') {
      spacer[0] += ' ';
      i--;
    }

    for (std::size_t k = 1; k < prefix.size(); k++) {
      int j = int(prefix[k].length()) - 1;
      while (j >= 0 && prefix[k][j] != '\n') {
        spacer[k] += ' ';
        j--;
      }
    }
  }

  std::string tenPrefix;
  std::string tenSuffix;
  std::vector<std::string> prefix;
  std::vector<std::string> suffix;
  std::vector<std::string> separator;
  char fill;
  int precision;
  int flags;
  std::vector<std::string> spacer{};
};

struct TensorIOFormatNumpy : public TensorIOFormatBase<TensorIOFormatNumpy> {
  using Base = TensorIOFormatBase<TensorIOFormatNumpy>;
  TensorIOFormatNumpy()
      : Base(/*separator=*/{" ", "\n"}, /*prefix=*/{"", "["}, /*suffix=*/{"", "]"}, /*precision=*/StreamPrecision,
             /*flags=*/0, /*tenPrefix=*/"[", /*tenSuffix=*/"]") {}
};

struct TensorIOFormatNative : public TensorIOFormatBase<TensorIOFormatNative> {
  using Base = TensorIOFormatBase<TensorIOFormatNative>;
  TensorIOFormatNative()
      : Base(/*separator=*/{", ", ",\n", "\n"}, /*prefix=*/{"", "{"}, /*suffix=*/{"", "}"},
             /*precision=*/StreamPrecision, /*flags=*/0, /*tenPrefix=*/"{", /*tenSuffix=*/"}") {}
};

struct TensorIOFormatPlain : public TensorIOFormatBase<TensorIOFormatPlain> {
  using Base = TensorIOFormatBase<TensorIOFormatPlain>;
  TensorIOFormatPlain()
      : Base(/*separator=*/{" ", "\n", "\n", ""}, /*prefix=*/{""}, /*suffix=*/{""}, /*precision=*/StreamPrecision,
             /*flags=*/0, /*tenPrefix=*/"", /*tenSuffix=*/"") {}
};

struct TensorIOFormatLegacy : public TensorIOFormatBase<TensorIOFormatLegacy> {
  using Base = TensorIOFormatBase<TensorIOFormatLegacy>;
  TensorIOFormatLegacy()
      : Base(/*separator=*/{", ", "\n"}, /*prefix=*/{"", "["}, /*suffix=*/{"", "]"}, /*precision=*/StreamPrecision,
             /*flags=*/0, /*tenPrefix=*/"", /*tenSuffix=*/"") {}
};

struct TensorIOFormat : public TensorIOFormatBase<TensorIOFormat> {
  using Base = TensorIOFormatBase<TensorIOFormat>;
  TensorIOFormat(const std::vector<std::string>& separator, const std::vector<std::string>& prefix,
                 const std::vector<std::string>& suffix, int precision = StreamPrecision, int flags = 0,
                 const std::string& tenPrefix = "", const std::string& tenSuffix = "", const char fill = ' ')
      : Base(separator, prefix, suffix, precision, flags, tenPrefix, tenSuffix, fill) {}

  static inline const TensorIOFormatNumpy Numpy() { return TensorIOFormatNumpy{}; }

  static inline const TensorIOFormatPlain Plain() { return TensorIOFormatPlain{}; }

  static inline const TensorIOFormatNative Native() { return TensorIOFormatNative{}; }

  static inline const TensorIOFormatLegacy Legacy() { return TensorIOFormatLegacy{}; }
};

template <typename T, int Layout, int rank, typename Format>
class TensorWithFormat;
// specialize for Layout=ColMajor, Layout=RowMajor and rank=0.
template <typename T, int rank, typename Format>
class TensorWithFormat<T, RowMajor, rank, Format> {
 public:
  TensorWithFormat(const T& tensor, const Format& format) : t_tensor(tensor), t_format(format) {}

  friend std::ostream& operator<<(std::ostream& os, const TensorWithFormat<T, RowMajor, rank, Format>& wf) {
    // Evaluate the expression if needed
    typedef TensorEvaluator<const TensorForcedEvalOp<const T>, DefaultDevice> Evaluator;
    TensorForcedEvalOp<const T> eval = wf.t_tensor.eval();
    Evaluator tensor(eval, DefaultDevice());
    tensor.evalSubExprsIfNeeded(NULL);
    internal::TensorPrinter<Evaluator, rank, Format>::run(os, tensor, wf.t_format);
    // Cleanup.
    tensor.cleanup();
    return os;
  }

 protected:
  T t_tensor;
  Format t_format;
};

template <typename T, int rank, typename Format>
class TensorWithFormat<T, ColMajor, rank, Format> {
 public:
  TensorWithFormat(const T& tensor, const Format& format) : t_tensor(tensor), t_format(format) {}

  friend std::ostream& operator<<(std::ostream& os, const TensorWithFormat<T, ColMajor, rank, Format>& wf) {
    // Switch to RowMajor storage and print afterwards
    typedef typename T::Index IndexType;
    std::array<IndexType, rank> shuffle;
    std::array<IndexType, rank> id;
    std::iota(id.begin(), id.end(), IndexType(0));
    std::copy(id.begin(), id.end(), shuffle.rbegin());
    auto tensor_row_major = wf.t_tensor.swap_layout().shuffle(shuffle);

    // Evaluate the expression if needed
    typedef TensorEvaluator<const TensorForcedEvalOp<const decltype(tensor_row_major)>, DefaultDevice> Evaluator;
    TensorForcedEvalOp<const decltype(tensor_row_major)> eval = tensor_row_major.eval();
    Evaluator tensor(eval, DefaultDevice());
    tensor.evalSubExprsIfNeeded(NULL);
    internal::TensorPrinter<Evaluator, rank, Format>::run(os, tensor, wf.t_format);
    // Cleanup.
    tensor.cleanup();
    return os;
  }

 protected:
  T t_tensor;
  Format t_format;
};

template <typename T, typename Format>
class TensorWithFormat<T, ColMajor, 0, Format> {
 public:
  TensorWithFormat(const T& tensor, const Format& format) : t_tensor(tensor), t_format(format) {}

  friend std::ostream& operator<<(std::ostream& os, const TensorWithFormat<T, ColMajor, 0, Format>& wf) {
    // Evaluate the expression if needed
    typedef TensorEvaluator<const TensorForcedEvalOp<const T>, DefaultDevice> Evaluator;
    TensorForcedEvalOp<const T> eval = wf.t_tensor.eval();
    Evaluator tensor(eval, DefaultDevice());
    tensor.evalSubExprsIfNeeded(NULL);
    internal::TensorPrinter<Evaluator, 0, Format>::run(os, tensor, wf.t_format);
    // Cleanup.
    tensor.cleanup();
    return os;
  }

 protected:
  T t_tensor;
  Format t_format;
};

namespace internal {

// Default scalar printer.
template <typename Scalar, typename Format, typename EnableIf = void>
struct ScalarPrinter {
  static void run(std::ostream& stream, const Scalar& scalar, const Format&) { stream << scalar; }
};

template <typename Scalar>
struct ScalarPrinter<Scalar, TensorIOFormatNumpy, std::enable_if_t<NumTraits<Scalar>::IsComplex>> {
  static void run(std::ostream& stream, const Scalar& scalar, const TensorIOFormatNumpy&) {
    stream << numext::real(scalar) << "+" << numext::imag(scalar) << "j";
  }
};

template <typename Scalar>
struct ScalarPrinter<Scalar, TensorIOFormatNative, std::enable_if_t<NumTraits<Scalar>::IsComplex>> {
  static void run(std::ostream& stream, const Scalar& scalar, const TensorIOFormatNative&) {
    stream << "{" << numext::real(scalar) << ", " << numext::imag(scalar) << "}";
  }
};

template <typename Tensor, std::size_t rank, typename Format, typename EnableIf>
struct TensorPrinter {
  using Scalar = std::remove_const_t<typename Tensor::Scalar>;

  static void run(std::ostream& s, const Tensor& tensor, const Format& fmt) {
    typedef typename Tensor::Index IndexType;

    eigen_assert(Tensor::Layout == RowMajor);
    typedef std::conditional_t<is_same<Scalar, char>::value || is_same<Scalar, unsigned char>::value ||
                                   is_same<Scalar, numext::int8_t>::value || is_same<Scalar, numext::uint8_t>::value,
                               int,
                               std::conditional_t<is_same<Scalar, std::complex<char>>::value ||
                                                      is_same<Scalar, std::complex<unsigned char>>::value ||
                                                      is_same<Scalar, std::complex<numext::int8_t>>::value ||
                                                      is_same<Scalar, std::complex<numext::uint8_t>>::value,
                                                  std::complex<int>, const Scalar&>>
        PrintType;

    const IndexType total_size = array_prod(tensor.dimensions());

    std::streamsize explicit_precision;
    if (fmt.precision == StreamPrecision) {
      explicit_precision = 0;
    } else if (fmt.precision == FullPrecision) {
      if (NumTraits<Scalar>::IsInteger) {
        explicit_precision = 0;
      } else {
        explicit_precision = significant_decimals_impl<Scalar>::run();
      }
    } else {
      explicit_precision = fmt.precision;
    }

    std::streamsize old_precision = 0;
    if (explicit_precision) old_precision = s.precision(explicit_precision);

    IndexType width = 0;
    bool align_cols = !(fmt.flags & DontAlignCols);
    if (align_cols) {
      // compute the largest width
      for (IndexType i = 0; i < total_size; i++) {
        std::stringstream sstr;
        sstr.copyfmt(s);
        ScalarPrinter<Scalar, Format>::run(sstr, static_cast<PrintType>(tensor.data()[i]), fmt);
        width = std::max<IndexType>(width, IndexType(sstr.str().length()));
      }
    }
    s << fmt.tenPrefix;
    for (IndexType i = 0; i < total_size; i++) {
      std::array<bool, rank> is_at_end{};
      std::array<bool, rank> is_at_begin{};

      // is the ith element the end of an coeff (always true), of a row, of a matrix, ...?
      for (std::size_t k = 0; k < rank; k++) {
        if ((i + 1) % (std::accumulate(tensor.dimensions().rbegin(), tensor.dimensions().rbegin() + k, 1,
                                       std::multiplies<IndexType>())) ==
            0) {
          is_at_end[k] = true;
        }
      }

      // is the ith element the begin of an coeff (always true), of a row, of a matrix, ...?
      for (std::size_t k = 0; k < rank; k++) {
        if (i % (std::accumulate(tensor.dimensions().rbegin(), tensor.dimensions().rbegin() + k, 1,
                                 std::multiplies<IndexType>())) ==
            0) {
          is_at_begin[k] = true;
        }
      }

      // do we have a line break?
      bool is_at_begin_after_newline = false;
      for (std::size_t k = 0; k < rank; k++) {
        if (is_at_begin[k]) {
          std::size_t separator_index = (k < fmt.separator.size()) ? k : fmt.separator.size() - 1;
          if (fmt.separator[separator_index].find('\n') != std::string::npos) {
            is_at_begin_after_newline = true;
          }
        }
      }

      bool is_at_end_before_newline = false;
      for (std::size_t k = 0; k < rank; k++) {
        if (is_at_end[k]) {
          std::size_t separator_index = (k < fmt.separator.size()) ? k : fmt.separator.size() - 1;
          if (fmt.separator[separator_index].find('\n') != std::string::npos) {
            is_at_end_before_newline = true;
          }
        }
      }

      std::stringstream suffix, prefix, separator;
      for (std::size_t k = 0; k < rank; k++) {
        std::size_t suffix_index = (k < fmt.suffix.size()) ? k : fmt.suffix.size() - 1;
        if (is_at_end[k]) {
          suffix << fmt.suffix[suffix_index];
        }
      }
      for (std::size_t k = 0; k < rank; k++) {
        std::size_t separator_index = (k < fmt.separator.size()) ? k : fmt.separator.size() - 1;
        if (is_at_end[k] &&
            (!is_at_end_before_newline || fmt.separator[separator_index].find('\n') != std::string::npos)) {
          separator << fmt.separator[separator_index];
        }
      }
      for (std::size_t k = 0; k < rank; k++) {
        std::size_t spacer_index = (k < fmt.spacer.size()) ? k : fmt.spacer.size() - 1;
        if (i != 0 && is_at_begin_after_newline && (!is_at_begin[k] || k == 0)) {
          prefix << fmt.spacer[spacer_index];
        }
      }
      for (int k = rank - 1; k >= 0; k--) {
        std::size_t prefix_index = (static_cast<std::size_t>(k) < fmt.prefix.size()) ? k : fmt.prefix.size() - 1;
        if (is_at_begin[k]) {
          prefix << fmt.prefix[prefix_index];
        }
      }

      s << prefix.str();
      // So we don't mess around with formatting, output scalar to a string stream, and adjust the width/fill manually.
      std::stringstream sstr;
      sstr.copyfmt(s);
      ScalarPrinter<Scalar, Format>::run(sstr, static_cast<PrintType>(tensor.data()[i]), fmt);
      std::string scalar_str = sstr.str();
      IndexType scalar_width = scalar_str.length();
      if (width && scalar_width < width) {
        std::string filler;
        for (IndexType j = scalar_width; j < width; ++j) {
          filler.push_back(fmt.fill);
        }
        s << filler;
      }
      s << scalar_str;
      s << suffix.str();
      if (i < total_size - 1) {
        s << separator.str();
      }
    }
    s << fmt.tenSuffix;
    if (explicit_precision) s.precision(old_precision);
  }
};

template <typename Tensor, std::size_t rank>
struct TensorPrinter<Tensor, rank, TensorIOFormatLegacy, std::enable_if_t<rank != 0>> {
  using Format = TensorIOFormatLegacy;
  using Scalar = std::remove_const_t<typename Tensor::Scalar>;

  static void run(std::ostream& s, const Tensor& tensor, const Format&) {
    typedef typename Tensor::Index IndexType;
    // backwards compatibility case: print tensor after reshaping to matrix of size dim(0) x
    // (dim(1)*dim(2)*...*dim(rank-1)).
    const IndexType total_size = internal::array_prod(tensor.dimensions());
    if (total_size > 0) {
      const IndexType first_dim = Eigen::internal::array_get<0>(tensor.dimensions());
      Map<const Array<Scalar, Dynamic, Dynamic, Tensor::Layout>> matrix(tensor.data(), first_dim,
                                                                        total_size / first_dim);
      s << matrix;
      return;
    }
  }
};

template <typename Tensor, typename Format>
struct TensorPrinter<Tensor, 0, Format> {
  static void run(std::ostream& s, const Tensor& tensor, const Format& fmt) {
    using Scalar = std::remove_const_t<typename Tensor::Scalar>;

    std::streamsize explicit_precision;
    if (fmt.precision == StreamPrecision) {
      explicit_precision = 0;
    } else if (fmt.precision == FullPrecision) {
      if (NumTraits<Scalar>::IsInteger) {
        explicit_precision = 0;
      } else {
        explicit_precision = significant_decimals_impl<Scalar>::run();
      }
    } else {
      explicit_precision = fmt.precision;
    }

    std::streamsize old_precision = 0;
    if (explicit_precision) old_precision = s.precision(explicit_precision);
    s << fmt.tenPrefix;
    ScalarPrinter<Scalar, Format>::run(s, tensor.coeff(0), fmt);
    s << fmt.tenSuffix;
    if (explicit_precision) s.precision(old_precision);
  }
};

}  // end namespace internal
template <typename T>
std::ostream& operator<<(std::ostream& s, const TensorBase<T, ReadOnlyAccessors>& t) {
  s << t.format(TensorIOFormat::Plain());
  return s;
}
}  // end namespace Eigen

#endif  // EIGEN_CXX11_TENSOR_TENSOR_IO_H
