// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <mkl_dfti.h>

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

#include <complex>
#include <memory>

namespace Eigen {
namespace internal {
namespace imklfft {

#define RUN_OR_ASSERT(EXPR, ERROR_MSG)                    \
  {                                                       \
    MKL_LONG status = (EXPR);                             \
    eigen_assert(status == DFTI_NO_ERROR && (ERROR_MSG)); \
  };

inline MKL_Complex16* complex_cast(const std::complex<double>* p) {
  return const_cast<MKL_Complex16*>(reinterpret_cast<const MKL_Complex16*>(p));
}

inline MKL_Complex8* complex_cast(const std::complex<float>* p) {
  return const_cast<MKL_Complex8*>(reinterpret_cast<const MKL_Complex8*>(p));
}

/*
 * Parameters:
 * precision: enum, Precision of the transform: DFTI_SINGLE or DFTI_DOUBLE.
 * forward_domain: enum, Forward domain of the transform: DFTI_COMPLEX or
 * DFTI_REAL. dimension: MKL_LONG Dimension of the transform. sizes: MKL_LONG if
 * dimension = 1.Length of the transform for a one-dimensional transform. sizes:
 * Array of type MKL_LONG otherwise. Lengths of each dimension for a
 * multi-dimensional transform.
 */
inline void configure_descriptor(std::shared_ptr<DFTI_DESCRIPTOR>& handl, enum DFTI_CONFIG_VALUE precision,
                                 enum DFTI_CONFIG_VALUE forward_domain, MKL_LONG dimension, MKL_LONG* sizes) {
  eigen_assert(dimension == 1 || dimension == 2 && "Transformation dimension must be less than 3.");

  DFTI_DESCRIPTOR_HANDLE res = nullptr;
  if (dimension == 1) {
    RUN_OR_ASSERT(DftiCreateDescriptor(&res, precision, forward_domain, dimension, *sizes),
                  "DftiCreateDescriptor failed.")
    handl.reset(res, [](DFTI_DESCRIPTOR_HANDLE handle) { DftiFreeDescriptor(&handle); });
    if (forward_domain == DFTI_REAL) {
      // Set CCE storage
      RUN_OR_ASSERT(DftiSetValue(handl.get(), DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX),
                    "DftiSetValue failed.")
    }
  } else {
    RUN_OR_ASSERT(DftiCreateDescriptor(&res, precision, DFTI_COMPLEX, dimension, sizes), "DftiCreateDescriptor failed.")
    handl.reset(res, [](DFTI_DESCRIPTOR_HANDLE handle) { DftiFreeDescriptor(&handle); });
  }

  RUN_OR_ASSERT(DftiSetValue(handl.get(), DFTI_PLACEMENT, DFTI_NOT_INPLACE), "DftiSetValue failed.")
  RUN_OR_ASSERT(DftiCommitDescriptor(handl.get()), "DftiCommitDescriptor failed.")
}

template <typename T>
struct plan {};

template <>
struct plan<float> {
  typedef float scalar_type;
  typedef MKL_Complex8 complex_type;

  std::shared_ptr<DFTI_DESCRIPTOR> m_plan;

  plan() = default;

  enum DFTI_CONFIG_VALUE precision = DFTI_SINGLE;

  inline void forward(complex_type* dst, complex_type* src, MKL_LONG nfft) {
    if (m_plan == 0) {
      configure_descriptor(m_plan, precision, DFTI_COMPLEX, 1, &nfft);
    }
    RUN_OR_ASSERT(DftiComputeForward(m_plan.get(), src, dst), "DftiComputeForward failed.")
  }

  inline void inverse(complex_type* dst, complex_type* src, MKL_LONG nfft) {
    if (m_plan == 0) {
      configure_descriptor(m_plan, precision, DFTI_COMPLEX, 1, &nfft);
    }
    RUN_OR_ASSERT(DftiComputeBackward(m_plan.get(), src, dst), "DftiComputeBackward failed.")
  }

  inline void forward(complex_type* dst, scalar_type* src, MKL_LONG nfft) {
    if (m_plan == 0) {
      configure_descriptor(m_plan, precision, DFTI_REAL, 1, &nfft);
    }
    RUN_OR_ASSERT(DftiComputeForward(m_plan.get(), src, dst), "DftiComputeForward failed.")
  }

  inline void inverse(scalar_type* dst, complex_type* src, MKL_LONG nfft) {
    if (m_plan == 0) {
      configure_descriptor(m_plan, precision, DFTI_REAL, 1, &nfft);
    }
    RUN_OR_ASSERT(DftiComputeBackward(m_plan.get(), src, dst), "DftiComputeBackward failed.")
  }

  inline void forward2(complex_type* dst, complex_type* src, int n0, int n1) {
    if (m_plan == 0) {
      MKL_LONG sizes[2] = {n0, n1};
      configure_descriptor(m_plan, precision, DFTI_COMPLEX, 2, sizes);
    }
    RUN_OR_ASSERT(DftiComputeForward(m_plan.get(), src, dst), "DftiComputeForward failed.")
  }

  inline void inverse2(complex_type* dst, complex_type* src, int n0, int n1) {
    if (m_plan == 0) {
      MKL_LONG sizes[2] = {n0, n1};
      configure_descriptor(m_plan, precision, DFTI_COMPLEX, 2, sizes);
    }
    RUN_OR_ASSERT(DftiComputeBackward(m_plan.get(), src, dst), "DftiComputeBackward failed.")
  }
};

template <>
struct plan<double> {
  typedef double scalar_type;
  typedef MKL_Complex16 complex_type;

  std::shared_ptr<DFTI_DESCRIPTOR> m_plan;

  plan() = default;

  enum DFTI_CONFIG_VALUE precision = DFTI_DOUBLE;

  inline void forward(complex_type* dst, complex_type* src, MKL_LONG nfft) {
    if (m_plan == 0) {
      configure_descriptor(m_plan, precision, DFTI_COMPLEX, 1, &nfft);
    }
    RUN_OR_ASSERT(DftiComputeForward(m_plan.get(), src, dst), "DftiComputeForward failed.")
  }

  inline void inverse(complex_type* dst, complex_type* src, MKL_LONG nfft) {
    if (m_plan == 0) {
      configure_descriptor(m_plan, precision, DFTI_COMPLEX, 1, &nfft);
    }
    RUN_OR_ASSERT(DftiComputeBackward(m_plan.get(), src, dst), "DftiComputeBackward failed.")
  }

  inline void forward(complex_type* dst, scalar_type* src, MKL_LONG nfft) {
    if (m_plan == 0) {
      configure_descriptor(m_plan, precision, DFTI_REAL, 1, &nfft);
    }
    RUN_OR_ASSERT(DftiComputeForward(m_plan.get(), src, dst), "DftiComputeForward failed.")
  }

  inline void inverse(scalar_type* dst, complex_type* src, MKL_LONG nfft) {
    if (m_plan == 0) {
      configure_descriptor(m_plan, precision, DFTI_REAL, 1, &nfft);
    }
    RUN_OR_ASSERT(DftiComputeBackward(m_plan.get(), src, dst), "DftiComputeBackward failed.")
  }

  inline void forward2(complex_type* dst, complex_type* src, int n0, int n1) {
    if (m_plan == 0) {
      MKL_LONG sizes[2] = {n0, n1};
      configure_descriptor(m_plan, precision, DFTI_COMPLEX, 2, sizes);
    }
    RUN_OR_ASSERT(DftiComputeForward(m_plan.get(), src, dst), "DftiComputeForward failed.")
  }

  inline void inverse2(complex_type* dst, complex_type* src, int n0, int n1) {
    if (m_plan == 0) {
      MKL_LONG sizes[2] = {n0, n1};
      configure_descriptor(m_plan, precision, DFTI_COMPLEX, 2, sizes);
    }
    RUN_OR_ASSERT(DftiComputeBackward(m_plan.get(), src, dst), "DftiComputeBackward failed.")
  }
};

template <typename Scalar_>
struct imklfft_impl {
  typedef Scalar_ Scalar;
  typedef std::complex<Scalar> Complex;

  inline void clear() { m_plans.clear(); }

  // complex-to-complex forward FFT
  inline void fwd(Complex* dst, const Complex* src, int nfft) {
    MKL_LONG size = nfft;
    get_plan(nfft, dst, src).forward(complex_cast(dst), complex_cast(src), size);
  }

  // real-to-complex forward FFT
  inline void fwd(Complex* dst, const Scalar* src, int nfft) {
    MKL_LONG size = nfft;
    get_plan(nfft, dst, src).forward(complex_cast(dst), const_cast<Scalar*>(src), nfft);
  }

  // 2-d complex-to-complex
  inline void fwd2(Complex* dst, const Complex* src, int n0, int n1) {
    get_plan(n0, n1, dst, src).forward2(complex_cast(dst), complex_cast(src), n0, n1);
  }

  // inverse complex-to-complex
  inline void inv(Complex* dst, const Complex* src, int nfft) {
    MKL_LONG size = nfft;
    get_plan(nfft, dst, src).inverse(complex_cast(dst), complex_cast(src), nfft);
  }

  // half-complex to scalar
  inline void inv(Scalar* dst, const Complex* src, int nfft) {
    MKL_LONG size = nfft;
    get_plan(nfft, dst, src).inverse(const_cast<Scalar*>(dst), complex_cast(src), nfft);
  }

  // 2-d complex-to-complex
  inline void inv2(Complex* dst, const Complex* src, int n0, int n1) {
    get_plan(n0, n1, dst, src).inverse2(complex_cast(dst), complex_cast(src), n0, n1);
  }

 private:
  std::map<int64_t, plan<Scalar>> m_plans;

  inline plan<Scalar>& get_plan(int nfft, void* dst, const void* src) {
    int inplace = dst == src ? 1 : 0;
    int aligned = ((reinterpret_cast<size_t>(src) & 15) | (reinterpret_cast<size_t>(dst) & 15)) == 0 ? 1 : 0;
    int64_t key = ((nfft << 2) | (inplace << 1) | aligned) << 1;

    // Create element if key does not exist.
    return m_plans[key];
  }

  inline plan<Scalar>& get_plan(int n0, int n1, void* dst, const void* src) {
    int inplace = (dst == src) ? 1 : 0;
    int aligned = ((reinterpret_cast<size_t>(src) & 15) | (reinterpret_cast<size_t>(dst) & 15)) == 0 ? 1 : 0;
    int64_t key = (((((int64_t)n0) << 31) | (n1 << 2) | (inplace << 1) | aligned) << 1) + 1;

    // Create element if key does not exist.
    return m_plans[key];
  }
};

#undef RUN_OR_ASSERT

}  // namespace imklfft
}  // namespace internal
}  // namespace Eigen
