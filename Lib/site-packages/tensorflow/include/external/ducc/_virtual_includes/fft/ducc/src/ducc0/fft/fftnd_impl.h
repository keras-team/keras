/*
This file is part of the ducc FFT library.

Copyright (C) 2010-2024 Max-Planck-Society
Copyright (C) 2019 Peter Bell

Authors: Martin Reinecke, Peter Bell
*/

/* SPDX-License-Identifier: BSD-3-Clause OR GPL-2.0-or-later */

/*
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice, this
  list of conditions and the following disclaimer in the documentation and/or
  other materials provided with the distribution.
* Neither the name of the copyright holder nor the names of its contributors may
  be used to endorse or promote products derived from this software without
  specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

/*
 *  This code is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This code is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this code; if not, write to the Free Software
 *  Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
 */

#ifndef DUCC0_FFTND_IMPL_H
#define DUCC0_FFTND_IMPL_H

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <numeric>
#include <stdexcept>
#include <memory>
#include <vector>
#include <complex>
#include <algorithm>
#include "ducc0/infra/useful_macros.h"
#include "ducc0/infra/error_handling.h"
#include "ducc0/infra/threading.h"
#include "ducc0/infra/misc_utils.h"
#include "ducc0/infra/simd.h"
#include "ducc0/infra/mav.h"
#include "ducc0/infra/aligned_array.h"
#include "ducc0/math/cmplx.h"
#include "ducc0/math/unity_roots.h"
#include "ducc0/fft/fft1d_impl.h"

/** \file fft.h
 *  Implementation of multi-dimensional Fast Fourier and related transforms
 *  \copyright Copyright (C) 2010-2021 Max-Planck-Society
 *  \copyright Copyright (C) 2019 Peter Bell
 *  \copyright
 *  \copyright For the odd-sized DCT-IV transforms:
 *  \copyright   Copyright (C) 2003, 2007-14 Matteo Frigo
 *  \copyright   Copyright (C) 2003, 2007-14 Massachusetts Institute of Technology
 *
 * \authors Martin Reinecke, Peter Bell
 */

namespace ducc0 {

namespace detail_fft {

// the next line is necessary to address some sloppy name choices in AdaptiveCpp
using std::min, std::max;

template<typename T> constexpr inline size_t fft_simdlen
  = min<size_t>(8, native_simd<T>::size());
template<> constexpr inline size_t fft_simdlen<double>
  = min<size_t>(4, native_simd<double>::size());
template<> constexpr inline size_t fft_simdlen<float>
  = min<size_t>(8, native_simd<float>::size());
template<typename T> using fft_simd = typename simd_select<T,fft_simdlen<T>>::type;
template<typename T> constexpr inline bool fft_simd_exists = (fft_simdlen<T> > 1);

struct util // hack to avoid duplicate symbols
  {
  static void sanity_check_axes(size_t ndim, const shape_t &axes)
    {
    if (ndim==1)
      {
      if ((axes.size()!=1) || (axes[0]!=0))
        throw std::invalid_argument("bad axes");
      return;
      }
    shape_t tmp(ndim,0);
    if (axes.empty()) throw std::invalid_argument("no axes specified");
    for (auto ax : axes)
      {
      if (ax>=ndim) throw std::invalid_argument("bad axis number");
      if (++tmp[ax]>1) throw std::invalid_argument("axis specified repeatedly");
      }
    }

  DUCC0_NOINLINE static void sanity_check_onetype(const fmav_info &a1,
    const fmav_info &a2, bool inplace, const shape_t &axes)
    {
    sanity_check_axes(a1.ndim(), axes);
    MR_assert(a1.conformable(a2), "array sizes are not conformable");
    if (inplace) MR_assert(a1.stride()==a2.stride(), "stride mismatch");
    }
  DUCC0_NOINLINE static void sanity_check_cr(const fmav_info &ac,
    const fmav_info &ar, const shape_t &axes)
    {
    sanity_check_axes(ac.ndim(), axes);
    MR_assert(ac.ndim()==ar.ndim(), "dimension mismatch");
    for (size_t i=0; i<ac.ndim(); ++i)
      MR_assert(ac.shape(i) == ((i==axes.back()) ? (ar.shape(i)/2+1) : ar.shape(i)),
        "axis length mismatch");
    }
  DUCC0_NOINLINE static void sanity_check_cr(const fmav_info &ac,
    const fmav_info &ar, const size_t axis)
    {
    if (axis>=ac.ndim()) throw std::invalid_argument("bad axis number");
    MR_assert(ac.ndim()==ar.ndim(), "dimension mismatch");
    for (size_t i=0; i<ac.ndim(); ++i)
      MR_assert(ac.shape(i) == ((i==axis) ? (ar.shape(i)/2+1) : ar.shape(i)),
        "axis length mismatch");
    }

  static size_t thread_count (size_t nthreads, const fmav_info &info,
    size_t axis, size_t /*vlen*/)
    {
    if (nthreads==1) return 1;
    size_t size = info.size();
    if (size<32768) return 1;  // not worth opening a parallel region
    size_t max_parallel = size / info.shape(axis);
    size_t max_threads = ducc0::adjust_nthreads(nthreads);
    return std::max(size_t(1), std::min(max_parallel, max_threads));
    }
  };

//
// multi-D infrastructure
//

template<typename T> std::shared_ptr<T> get_plan(size_t length, bool vectorize=false)
  {
#ifdef DUCC0_NO_FFT_CACHE
  return std::make_shared<T>(length, vectorize);
#else
  constexpr size_t nmax=10;
  struct entry { size_t n; bool vectorize; std::shared_ptr<T> ptr; };
  static std::array<entry, nmax> cache{{{0,0,nullptr}}};
  static std::array<size_t, nmax> last_access{{0}};
  static size_t access_counter = 0;
  static Mutex mut;

  auto find_in_cache = [&]() -> std::shared_ptr<T>
    {
    for (size_t i=0; i<nmax; ++i)
      if (cache[i].ptr && (cache[i].n==length) && (cache[i].vectorize==vectorize))
        {
        // no need to update if this is already the most recent entry
        if (last_access[i]!=access_counter)
          {
          last_access[i] = ++access_counter;
          // Guard against overflow
          if (access_counter == 0)
            last_access.fill(0);
          }
        return cache[i].ptr;
        }

    return nullptr;
    };

  {
  LockGuard lock(mut);

  auto p = find_in_cache();
  if (p) return p;
  }
  auto plan = std::make_shared<T>(length, vectorize);
  {
  LockGuard lock(mut);

  auto p = find_in_cache();
  if (p) return p;

  size_t lru = 0;
  for (size_t i=1; i<nmax; ++i)
    if (last_access[i] < last_access[lru])
      lru = i;

  cache[lru] = {length,vectorize, plan};
  last_access[lru] = ++access_counter;
  }
  return plan;
#endif
  }

template<size_t N> class multi_iter
  {
  private:
    shape_t shp, pos;
    stride_t str_i, str_o;
    size_t cshp_i, cshp_o, rem;
    ptrdiff_t cstr_i, cstr_o, sstr_i, sstr_o, p_ii, p_i[N], p_oi, p_o[N];
    bool uni_i, uni_o;

    void advance_i()
      {
      for (size_t i=0; i<pos.size(); ++i)
        {
        p_ii += str_i[i];
        p_oi += str_o[i];
        if (++pos[i] < shp[i])
          return;
        pos[i] = 0;
        p_ii -= ptrdiff_t(shp[i])*str_i[i];
        p_oi -= ptrdiff_t(shp[i])*str_o[i];
        }
      }

  public:
    multi_iter(const fmav_info &iarr, const fmav_info &oarr, size_t idim,
      size_t nshares, size_t myshare)
      : rem(iarr.size()/iarr.shape(idim)), sstr_i(0), sstr_o(0), p_ii(0), p_oi(0)
      {
      MR_assert(oarr.ndim()==iarr.ndim(), "dimension mismatch");
      MR_assert(iarr.ndim()>=1, "not enough dimensions");
      // Sort the extraneous dimensions in order of ascending output stride;
      // this should improve overall cache re-use and avoid clashes between
      // threads as much as possible.
      shape_t idx(iarr.ndim());
      std::iota(idx.begin(), idx.end(), 0);
      sort(idx.begin(), idx.end(),
        [&oarr](size_t i1, size_t i2) {return oarr.stride(i1) < oarr.stride(i2);});
      for (auto i: idx)
        if (i!=idim)
          {
          pos.push_back(0);
          MR_assert(iarr.shape(i)==oarr.shape(i), "shape mismatch");
          shp.push_back(iarr.shape(i));
          str_i.push_back(iarr.stride(i));
          str_o.push_back(oarr.stride(i));
          }
      MR_assert(idim<iarr.ndim(), "bad active dimension");
      cstr_i = iarr.stride(idim);
      cstr_o = oarr.stride(idim);
      cshp_i = iarr.shape(idim);
      cshp_o = oarr.shape(idim);

// collapse unneeded dimensions
      bool done = false;
      while(!done)
        {
        done=true;
        for (size_t i=1; i<shp.size(); ++i)
          if ((str_i[i] == str_i[i-1]*ptrdiff_t(shp[i-1]))
           && (str_o[i] == str_o[i-1]*ptrdiff_t(shp[i-1])))
            {
            shp[i-1] *= shp[i];
            str_i.erase(str_i.begin()+ptrdiff_t(i));
            str_o.erase(str_o.begin()+ptrdiff_t(i));
            shp.erase(shp.begin()+ptrdiff_t(i));
            pos.pop_back();
            done=false;
            }
        }
      if (pos.size()>0)
        {
        sstr_i = str_i[0];
        sstr_o = str_o[0];
        }

      if (nshares==1) return;
      if (nshares==0) throw std::runtime_error("can't run with zero threads");
      if (myshare>=nshares) throw std::runtime_error("impossible share requested");
      auto [lo, hi] = calcShare(nshares, myshare, rem);
      size_t todo = hi-lo;

      // this shouldn't normally happen, but make sure we handle it properly
      if (todo==0)
        { rem=0; return; }

      size_t chunk = rem;
      for (size_t i2=0, i=pos.size()-1; i2<pos.size(); ++i2,--i)
        {
        chunk /= shp[i];
        size_t n_advance = lo/chunk;
        pos[i] += n_advance;
        p_ii += ptrdiff_t(n_advance)*str_i[i];
        p_oi += ptrdiff_t(n_advance)*str_o[i];
        lo -= n_advance*chunk;
        }
      MR_assert(lo==0, "must not happen");
      rem = todo;
      }
    void advance(size_t n)
      {
      if (rem<n) throw std::runtime_error("underrun");
      for (size_t i=0; i<n; ++i)
        {
        p_i[i] = p_ii;
        p_o[i] = p_oi;
        advance_i();
        }
      uni_i = uni_o = true;
      for (size_t i=1; i<n; ++i)
        {
        uni_i = uni_i && (p_i[i]-p_i[i-1] == sstr_i);
        uni_o = uni_o && (p_o[i]-p_o[i-1] == sstr_o);
        }
      rem -= n;
      }
    ptrdiff_t iofs(size_t i) const { return p_i[0] + ptrdiff_t(i)*cstr_i; }
    ptrdiff_t iofs(size_t j, size_t i) const { return p_i[j] + ptrdiff_t(i)*cstr_i; }
    ptrdiff_t iofs_uni(size_t j, size_t i) const { return p_i[0] + ptrdiff_t(j)*sstr_i + ptrdiff_t(i)*cstr_i; }
    ptrdiff_t oofs(size_t i) const { return p_o[0] + ptrdiff_t(i)*cstr_o; }
    ptrdiff_t oofs(size_t j, size_t i) const { return p_o[j] + ptrdiff_t(i)*cstr_o; }
    ptrdiff_t oofs_uni(size_t j, size_t i) const { return p_o[0] + ptrdiff_t(j)*sstr_o + ptrdiff_t(i)*cstr_o; }
    bool uniform_i() const { return uni_i; }
    ptrdiff_t unistride_i() const { return sstr_i; }
    bool uniform_o() const { return uni_o; }
    ptrdiff_t unistride_o() const { return sstr_o; }
    size_t length_in() const { return cshp_i; }
    size_t length_out() const { return cshp_o; }
    ptrdiff_t stride_in() const { return cstr_i; }
    ptrdiff_t stride_out() const { return cstr_o; }
    size_t remaining() const { return rem; }
    bool critical_stride_trans(size_t tsz) const
      {
      return ((abs<ptrdiff_t>(stride_in() *tsz)&4095)==0)
          || ((abs<ptrdiff_t>(stride_out()*tsz)&4095)==0);
      }
    bool critical_stride_other(size_t tsz) const
      {
      if (unistride_i()==0) return false;  // it's just one transform
      return ((abs<ptrdiff_t>(unistride_i()*tsz)&4095)==0)
          || ((abs<ptrdiff_t>(unistride_o()*tsz)&4095)==0);
      }
  };

template<typename T, typename T0> class TmpStorage
  {
  private:
    aligned_array<T> d;
    size_t dofs, dstride;

  public:
    TmpStorage(size_t n_trafo, size_t bufsize_data, size_t bufsize_trafo,
               size_t n_simultaneous, bool inplace)
      {
      if (inplace)
        {
        d.resize(bufsize_trafo);
        return;
        }
      constexpr auto vlen = fft_simdlen<T0>;
      // FIXME: when switching to C++20, use bit_floor(othersize)
      size_t buffct = std::min(vlen, n_trafo);
      size_t datafct = std::min(vlen, n_trafo);
      if (n_trafo>=n_simultaneous*vlen) datafct = n_simultaneous*vlen;
      dstride = bufsize_data;
      dofs = bufsize_trafo;
      // critical stride avoidance
      if ((dstride&256)==0) dstride+=16;
      if ((dofs&256)==0) dofs += 16;
      d.resize(buffct*dofs + datafct*dstride);
      }

    template<typename T2> T2 *transformBuf()
      { return reinterpret_cast<T2 *>(d.data()); }
    template<typename T2> T2 *dataBuf()
      { return reinterpret_cast<T2 *>(d.data()) + dofs; }
    size_t data_stride() const
      { return dstride; }
  };

template<typename T2, typename T, typename T0> class TmpStorage2
  {
  private:
    TmpStorage<T, T0> &stg;

  public:
    using datatype = T2;
    TmpStorage2(TmpStorage<T,T0> &stg_): stg(stg_) {}

    T2 *transformBuf() { return stg.template transformBuf<T2>(); }
    T2 *dataBuf() { return stg.template dataBuf<T2>(); }
    size_t data_stride() const { return stg.data_stride(); }
  };

template <typename Tsimd, typename Titer> DUCC0_NOINLINE void copy_input(const Titer &it,
  const cfmav<Cmplx<typename Tsimd::value_type>> &src, Cmplx<Tsimd> *DUCC0_RESTRICT dst)
  {
  constexpr auto vlen=Tsimd::size();
  const Cmplx<typename Tsimd::value_type> * DUCC0_RESTRICT ptr = src.data();
  for (size_t i=0; i<it.length_in(); ++i)
    {
    Cmplx<Tsimd> tmp;
    for (size_t j=0; j<vlen; ++j)
      {
      tmp.r[j] = ptr[it.iofs(j,i)].r;
      tmp.i[j] = ptr[it.iofs(j,i)].i;
      }
    dst[i] = tmp;
    }
  }

template <typename Tsimd, typename Titer> DUCC0_NOINLINE void copy_input(const Titer &it,
  const cfmav<typename Tsimd::value_type> &src, Tsimd *DUCC0_RESTRICT dst)
  {
  constexpr auto vlen=Tsimd::size();
  const typename Tsimd::value_type * DUCC0_RESTRICT ptr = src.data();
  for (size_t i=0; i<it.length_in(); ++i)
    {
    typename Tsimd::value_type tmp[vlen];
    for (size_t j=0; j<vlen; ++j)
      tmp[j] = ptr[it.iofs(j,i)];
    dst[i] = Tsimd(&tmp[0], element_aligned_tag());
    }
  }

template <typename Titer, typename T> DUCC0_NOINLINE void copy_input(const Titer &it,
  const cfmav<T> &src, T *DUCC0_RESTRICT dst)
  {
  const T * DUCC0_RESTRICT ptr = src.data();
  if (dst == &src.raw(it.iofs(0))) return;  // in-place
  for (size_t i=0; i<it.length_in(); ++i)
    dst[i] = ptr[it.iofs(i)];
  }

template<typename Tsimd, typename Titer> DUCC0_NOINLINE void copy_output(const Titer &it,
  const Cmplx<Tsimd> *DUCC0_RESTRICT src, const vfmav<Cmplx<typename Tsimd::value_type>> &dst)
  {
  constexpr auto vlen=Tsimd::size();
  Cmplx<typename Tsimd::value_type> * DUCC0_RESTRICT ptr = dst.data();
  for (size_t i=0; i<it.length_out(); ++i)
    {
    Cmplx<Tsimd> tmp(src[i]);
    for (size_t j=0; j<vlen; ++j)
      ptr[it.oofs(j,i)].Set(tmp.r[j],tmp.i[j]);
    }
  }

template<typename Tsimd, typename Titer> DUCC0_NOINLINE void copy_output(const Titer &it,
  const Tsimd *DUCC0_RESTRICT src, const vfmav<typename Tsimd::value_type> &dst)
  {
  constexpr auto vlen=Tsimd::size();
  typename Tsimd::value_type * DUCC0_RESTRICT ptr = dst.data();
  for (size_t i=0; i<it.length_out(); ++i)
    {
    Tsimd tmp = src[i];
    for (size_t j=0; j<vlen; ++j)
      ptr[it.oofs(j,i)] = tmp[j];
    }
  }

template<typename T, typename Titer> DUCC0_NOINLINE void copy_output(const Titer &it,
  const T *DUCC0_RESTRICT src, const vfmav<T> &dst)
  {
  T * DUCC0_RESTRICT ptr=dst.data();
  if (src == &dst.raw(it.oofs(0))) return;  // in-place
  for (size_t i=0; i<it.length_out(); ++i)
    ptr[it.oofs(i)] = src[i];
  }
template <typename Tsimd, typename Titer> DUCC0_NOINLINE void copy_input(const Titer &it,
  const cfmav<Cmplx<typename Tsimd::value_type>> &src, Cmplx<Tsimd> * DUCC0_RESTRICT dst, size_t nvec, size_t vstr)
  {
  constexpr auto vlen=Tsimd::size();
  const Cmplx<typename Tsimd::value_type> * DUCC0_RESTRICT ptr = src.data();
  for (size_t i=0; i<it.length_in(); ++i)
    for (size_t j0=0; j0<nvec; ++j0)
      {
      typename Tsimd::value_type tmp[2*vlen];
      for (size_t j1=0; j1<vlen; ++j1)
        {
        tmp[j1] = ptr[it.iofs(j0*vlen+j1,i)].r;
        tmp[j1+vlen] = ptr[it.iofs(j0*vlen+j1,i)].i;
        }

      dst[j0*vstr+i].r = Tsimd(&tmp[0], element_aligned_tag());
      dst[j0*vstr+i].i = Tsimd(&tmp[vlen], element_aligned_tag());
      }
  }
template <typename T, typename Titer> DUCC0_NOINLINE void copy_input(const Titer &it,
  const cfmav<Cmplx<T>> &src, Cmplx<T> * DUCC0_RESTRICT dst, size_t nvec, size_t vstr)
  {
  const Cmplx<T> * DUCC0_RESTRICT ptr = src.data();
  for (size_t i=0; i<it.length_in(); ++i)
    for (size_t j0=0; j0<nvec; ++j0)
      dst[j0*vstr+i] = ptr[it.iofs(j0,i)];
  }

template <typename Tsimd, typename Titer> DUCC0_NOINLINE void copy_input(const Titer &it,
  const cfmav<typename Tsimd::value_type> &src, Tsimd * DUCC0_RESTRICT dst, size_t nvec, size_t vstr)
  {
  constexpr auto vlen=Tsimd::size();
  const typename Tsimd::value_type * DUCC0_RESTRICT ptr = src.data();
  for (size_t i=0; i<it.length_in(); ++i)
    for (size_t j0=0; j0<nvec; ++j0)
      {
      typename Tsimd::value_type tmp[vlen];
      for (size_t j1=0; j1<vlen; ++j1)
        tmp[j1] = ptr[it.iofs(j0*vlen+j1,i)];
      dst[j0*vstr+i] = Tsimd(&tmp[0],element_aligned_tag());
      }
  }

template <typename T, typename Titer> DUCC0_NOINLINE void copy_input(const Titer &it,
  const cfmav<T> &src, T * DUCC0_RESTRICT dst, size_t nvec, size_t vstr)
  {
  const T * DUCC0_RESTRICT ptr = src.data();
  for (size_t i=0; i<it.length_in(); ++i)
    for (size_t j0=0; j0<nvec; ++j0)
      dst[j0*vstr+i] = ptr[it.iofs(j0,i)];
  }

template<typename Tsimd, typename Titer> DUCC0_NOINLINE void copy_output(const Titer &it,
  const Cmplx<Tsimd> * DUCC0_RESTRICT src, const vfmav<Cmplx<typename Tsimd::value_type>> &dst, size_t nvec, size_t vstr)
  {
  constexpr auto vlen=Tsimd::size();
  Cmplx<typename Tsimd::value_type> * DUCC0_RESTRICT ptr = dst.data();
  for (size_t i=0; i<it.length_out(); ++i)
    for (size_t j0=0; j0<nvec; ++j0)
      {
      Cmplx<Tsimd> tmp(src[j0*vstr+i]);
      for (size_t j1=0; j1<vlen; ++j1)
        ptr[it.oofs(j0*vlen+j1,i)].Set(tmp.r[j1],tmp.i[j1]);
      }
  }
template<typename T, typename Titer> DUCC0_NOINLINE void copy_output(const Titer &it,
  const Cmplx<T> * DUCC0_RESTRICT src, const vfmav<Cmplx<T>> &dst, size_t nvec, size_t vstr)
  {
  Cmplx<T> * DUCC0_RESTRICT ptr = dst.data();
  for (size_t i=0; i<it.length_out(); ++i)
    for (size_t j0=0; j0<nvec; ++j0)
      ptr[it.oofs(j0,i)] = src[j0*vstr+i];
  }
template<typename Tsimd, typename Titer> DUCC0_NOINLINE void copy_output(const Titer &it,
  const Tsimd * DUCC0_RESTRICT src, const vfmav<typename Tsimd::value_type> &dst, size_t nvec, size_t vstr)
  {
  constexpr auto vlen=Tsimd::size();
  typename Tsimd::value_type * DUCC0_RESTRICT ptr = dst.data();
  for (size_t i=0; i<it.length_out(); ++i)
    for (size_t j0=0; j0<nvec; ++j0)
      {
      Tsimd tmp(src[j0*vstr+i]);
      for (size_t j1=0; j1<vlen; ++j1)
        ptr[it.oofs(j0*vlen+j1,i)] = tmp[j1];
      }
  }
template<typename T, typename Titer> DUCC0_NOINLINE void copy_output(const Titer &it,
  const T * DUCC0_RESTRICT src, const vfmav<T> &dst, size_t nvec, size_t vstr)
  {
  T * DUCC0_RESTRICT ptr = dst.data();
  for (size_t i=0; i<it.length_out(); ++i)
    for (size_t j0=0; j0<nvec; ++j0)
      ptr[it.oofs(j0,i)] = src[j0*vstr+i];
  }


template <typename T, size_t vlen> struct add_vec
  { using type = typename simd_select<T, vlen>::type; };
template <typename T, size_t vlen> struct add_vec<Cmplx<T>, vlen>
  { using type = Cmplx<typename simd_select<T, vlen>::type>; };
template <typename T, size_t vlen> using add_vec_t = typename add_vec<T, vlen>::type;

template<typename Tplan, typename T, typename T0, typename Exec>
DUCC0_NOINLINE void general_nd(const cfmav<T> &in, const vfmav<T> &out,
  const shape_t &axes, T0 fct, size_t nthreads, const Exec &exec,
  const bool /*allow_inplace*/=true)
  {
  if ((in.ndim()==1)&&(in.stride(0)==1)&&(out.stride(0)==1))
    {
    auto plan = get_plan<Tplan>(in.shape(0), true);
    exec.exec_simple(in.data(), out.data(), *plan, fct, nthreads);
    return;
    }
  std::shared_ptr<Tplan> plan, vplan;
  size_t nth1d = (in.ndim()==1) ? nthreads : 1;

  for (size_t iax=0; iax<axes.size(); ++iax)
    {
    size_t len=in.shape(axes[iax]);
    if ((!plan) || (len!=plan->length()))
      {
      plan = get_plan<Tplan>(len, in.ndim()==1);
      vplan = ((in.ndim()==1)||(len<300)||((len&3)!=0)) ?
        plan : get_plan<Tplan>(len, true);
      }

    execParallel(util::thread_count(nthreads, in, axes[iax], fft_simdlen<T0>),
      [&](Scheduler &sched)
      {
      constexpr auto vlen = fft_simdlen<T0>;
      constexpr size_t nmax = 16;
      const auto &tin(iax==0? in : out);
      multi_iter<nmax> it(tin, out, axes[iax], sched.num_threads(), sched.thread_num());

      // n_simul: vector size
      // n_bunch: total size of bunch (multiple of n_simul)
      size_t n_simul=1, n_bunch=1;
      bool critstride = (((in.stride(axes[iax])*sizeof(T))&4095)==0)
                     || (((out.stride(axes[iax])*sizeof(T))&4095)==0);
      bool nostride = (in.stride(axes[iax])==1) && (out.stride(axes[iax])==1);

      constexpr size_t l2cache=262144*2;
      constexpr size_t cacheline=64;

      // working set size
      auto wss = [&](size_t vl) { return sizeof(T)*(2*len*vl + plan->bufsize()); };
      // is the FFT small enough to fit into L2 vectorized?
      if (wss(1)>l2cache) // "long" FFT, don't execute more than one at the same time
        {
        n_simul=1;
        if (critstride)  // make bunch large to reduce overall copy cost
          {
          n_bunch=n_simul;
          while ((n_bunch<nmax) && (sizeof(T)*n_bunch<2*cacheline)) n_bunch*=2;
          }
        else if (nostride)  // simple scalar "in-place" transform
          n_bunch=n_simul;
        else  // we have some strides, use a medium-sized bunch
          {
          n_bunch=n_simul;
          while ((n_bunch<nmax) && (sizeof(T)*n_bunch<cacheline)) n_bunch*=2;
          }
        }
      else  // fairly small individual FFT, vectorizing probably beneficial
        {
        // if no stride, only vectorize if vectorized FFT fits into cache
        // if strided, always vectorize (TBC)
        n_simul = nostride ? ((wss(vlen)<=l2cache) ? vlen:1) : vlen;
        if (critstride)  // make bunch large to reduce overall copy cost
          {
          n_bunch=n_simul;
          while ((n_bunch<nmax) /*&& (sizeof(T)*n_bunch<2*cacheline)*/) n_bunch*=2;
          }
        else if (nostride)
          n_bunch=n_simul;
        else
          {
          n_bunch=n_simul;
          if (n_simul==1)
            while ((n_bunch<nmax) && (sizeof(T)*n_bunch<cacheline)) n_bunch*=2;
          }
        }

      bool inplace = (in.stride(axes[iax])==1) && (out.stride(axes[iax])==1) && (n_bunch==1);
      MR_assert(n_bunch<=nmax, "must not happen");
      TmpStorage<T,T0> storage(in.size()/len, len, max(plan->bufsize(),vplan->bufsize()), (n_bunch+vlen-1)/vlen, inplace);

      // first, do all possible steps of size n_bunch, then n_simul
      if (n_bunch>1)
        {
#ifndef DUCC0_NO_SIMD
        if constexpr (vlen>1)
          {
          constexpr size_t lvlen = vlen;
          if (n_simul>=lvlen)
            {
            if ((n_bunch>n_simul) && (it.remaining()>=n_bunch))
              {
              TmpStorage2<add_vec_t<T, lvlen>,T,T0> storage2(storage);
              while (it.remaining()>=n_bunch)
                {
                it.advance(n_bunch);
                exec.exec_n(it, tin, out, storage2, *plan, fct, n_bunch/lvlen, nth1d);
                }
              }
            }
          if (n_simul==lvlen)
            {
            if (it.remaining()>=lvlen)
              {
              TmpStorage2<add_vec_t<T, lvlen>,T,T0> storage2(storage);
              while (it.remaining()>=lvlen)
                {
                it.advance(lvlen);
                exec(it, tin, out, storage2, *plan, fct, nth1d);
                }
              }
            }
          }
        if constexpr ((vlen>2) && (simd_exists<T0,vlen/2>))
          {
          constexpr size_t lvlen = vlen/2;
          if (n_simul>=lvlen)
            {
            if ((n_bunch>n_simul) && (it.remaining()>=n_bunch))
              {
              TmpStorage2<add_vec_t<T, lvlen>,T,T0> storage2(storage);
              while (it.remaining()>=n_bunch)
                {
                it.advance(n_bunch);
                exec.exec_n(it, tin, out, storage2, *plan, fct, n_bunch/lvlen, nth1d);
                }
              }
            }
          if (n_simul==lvlen)
            {
            if (it.remaining()>=lvlen)
              {
              TmpStorage2<add_vec_t<T, lvlen>,T,T0> storage2(storage);
              while (it.remaining()>=lvlen)
                {
                it.advance(lvlen);
                exec(it, tin, out, storage2, *plan, fct, nth1d);
                }
              }
            }
          }
        if constexpr ((vlen>4) && (simd_exists<T0,vlen/4>))
          {
          constexpr size_t lvlen = vlen/4;
          if (n_simul>=lvlen)
            {
            if ((n_bunch>n_simul) && (it.remaining()>=n_bunch))
              {
              TmpStorage2<add_vec_t<T, lvlen>,T,T0> storage2(storage);
              while (it.remaining()>=n_bunch)
                {
                it.advance(n_bunch);
                exec.exec_n(it, tin, out, storage2, *plan, fct, n_bunch/lvlen, nth1d);
                }
              }
            }
          if (n_simul==lvlen)
            {
            if (it.remaining()>=lvlen)
              {
              TmpStorage2<add_vec_t<T, lvlen>,T,T0> storage2(storage);
              while (it.remaining()>=lvlen)
                {
                it.advance(lvlen);
                exec(it, tin, out, storage2, *plan, fct, nth1d);
                }
              }
            }
          }
#endif
        {
        TmpStorage2<T,T,T0> storage2(storage);
        while ((n_bunch>n_simul) && (it.remaining()>=n_bunch))
          {
          it.advance(n_bunch);
          exec.exec_n(it, tin, out, storage2, *vplan, fct, n_bunch, nth1d);
          }
        }
        }
        {
        TmpStorage2<T,T,T0> storage2(storage);
        while (it.remaining()>0)
          {
          it.advance(1);
          exec(it, tin, out, storage2, *vplan, fct, nth1d, inplace);
          }
        }
      });  // end of parallel region
    fct = T0(1); // factor has been applied, use 1 for remaining axes
    }
  }

struct ExecC2C
  {
  bool forward;

  template <typename T0, typename Tstorage, typename Titer> DUCC0_NOINLINE void operator() (
    const Titer &it, const cfmav<Cmplx<T0>> &in,
    const vfmav<Cmplx<T0>> &out, Tstorage &storage, const pocketfft_c<T0> &plan, T0 fct,
    size_t nthreads, bool inplace=false) const
    {
    using T = typename Tstorage::datatype;
    if constexpr(is_same<Cmplx<T0>, T>::value)
      if (inplace)
        {
        if (in.data()!=out.data())
          copy_input(it, in, out.data()+it.oofs(0));
        plan.exec_copyback(out.data()+it.oofs(0), storage.transformBuf(), fct, forward, nthreads);
        return;
        }
    T *buf1=storage.transformBuf(), *buf2=storage.dataBuf();
    copy_input(it, in, buf2);
    auto res = plan.exec(buf2, buf1, fct, forward, nthreads);
    copy_output(it, res, out);
    }
  template <typename T0, typename Tstorage, typename Titer> DUCC0_NOINLINE void exec_n (
    const Titer &it, const cfmav<Cmplx<T0>> &in,
    const vfmav<Cmplx<T0>> &out, Tstorage &storage, const pocketfft_c<T0> &plan, T0 fct, size_t nvec,
    size_t nthreads) const
    {
    using T = typename Tstorage::datatype;
    size_t dstr = storage.data_stride();
    T *buf1=storage.transformBuf(), *buf2=storage.dataBuf();
    copy_input(it, in, buf2, nvec, dstr);
    for (size_t i=0; i<nvec; ++i)
      plan.exec_copyback(buf2+i*dstr, buf1, fct, forward, nthreads);
    copy_output(it, buf2, out, nvec, dstr);
    }
  template <typename T0> DUCC0_NOINLINE void exec_simple (
    const Cmplx<T0> *in, Cmplx<T0> *out, const pocketfft_c<T0> &plan, T0 fct,
    size_t nthreads) const
    {
    if (in!=out) copy_n(in, plan.length(), out);
    plan.exec(out, fct, forward, nthreads);
    }
  };

struct ExecHartley
  {
  template <typename T0, typename Tstorage, typename Titer> DUCC0_NOINLINE void operator() (
    const Titer &it, const cfmav<T0> &in, const vfmav<T0> &out,
    Tstorage &storage, const pocketfft_hartley<T0> &plan, T0 fct, size_t nthreads,
    bool inplace=false) const
    {
    using T = typename Tstorage::datatype;
    if constexpr(is_same<T0, T>::value)
      if (inplace)
        {
        if (in.data()!=out.data())
          copy_input(it, in, out.data()+it.oofs(0));
        plan.exec_copyback(out.data()+it.oofs(0), storage.transformBuf(), fct, nthreads);
        return;
        }
    T *buf1=storage.transformBuf(), *buf2=storage.dataBuf();
    copy_input(it, in, buf2);
    auto res = plan.exec(buf2, buf1, fct, nthreads);
    copy_output(it, res, out);
    }
  template <typename T0, typename Tstorage, typename Titer> DUCC0_NOINLINE void exec_n (
    const Titer &it, const cfmav<T0> &in,
    const vfmav<T0> &out, Tstorage &storage, const pocketfft_hartley<T0> &plan, T0 fct, size_t nvec,
    size_t nthreads) const
    {
    using T = typename Tstorage::datatype;
    size_t dstr = storage.data_stride();
    T *buf1=storage.transformBuf(), *buf2=storage.dataBuf();
    copy_input(it, in, buf2, nvec, dstr);
    for (size_t i=0; i<nvec; ++i)
      plan.exec_copyback(buf2+i*dstr, buf1, fct, nthreads);
    copy_output(it, buf2, out, nvec, dstr);
    }
  template <typename T0> DUCC0_NOINLINE void exec_simple (
    const T0 *in, T0 *out, const pocketfft_hartley<T0> &plan, T0 fct,
    size_t nthreads) const
    {
    if (in!=out) copy_n(in, plan.length(), out);
    plan.exec(out, fct, nthreads);
    }
  };

struct ExecFHT
  {
  template <typename T0, typename Tstorage, typename Titer> DUCC0_NOINLINE void operator() (
    const Titer &it, const cfmav<T0> &in, const vfmav<T0> &out,
    Tstorage &storage, const pocketfft_fht<T0> &plan, T0 fct, size_t nthreads,
    bool inplace=false) const
    {
    using T = typename Tstorage::datatype;
    if constexpr(is_same<T0, T>::value)
      if (inplace)
        {
        if (in.data()!=out.data())
          copy_input(it, in, out.data()+it.oofs(0));
        plan.exec_copyback(out.data()+it.oofs(0), storage.transformBuf(), fct, nthreads);
        return;
        }
    T *buf1=storage.transformBuf(), *buf2=storage.dataBuf();
    copy_input(it, in, buf2);
    auto res = plan.exec(buf2, buf1, fct, nthreads);
    copy_output(it, res, out);
    }
  template <typename T0, typename Tstorage, typename Titer> DUCC0_NOINLINE void exec_n (
    const Titer &it, const cfmav<T0> &in,
    const vfmav<T0> &out, Tstorage &storage, const pocketfft_fht<T0> &plan, T0 fct, size_t nvec,
    size_t nthreads) const
    {
    using T = typename Tstorage::datatype;
    size_t dstr = storage.data_stride();
    T *buf1=storage.transformBuf(), *buf2=storage.dataBuf();
    copy_input(it, in, buf2, nvec, dstr);
    for (size_t i=0; i<nvec; ++i)
      plan.exec_copyback(buf2+i*dstr, buf1, fct, nthreads);
    copy_output(it, buf2, out, nvec, dstr);
    }
  template <typename T0> DUCC0_NOINLINE void exec_simple (
    const T0 *in, T0 *out, const pocketfft_fht<T0> &plan, T0 fct,
    size_t nthreads) const
    {
    if (in!=out) copy_n(in, plan.length(), out);
    plan.exec(out, fct, nthreads);
    }
  };

struct ExecFFTW
  {
  bool forward;

  template <typename T0, typename Tstorage, typename Titer> DUCC0_NOINLINE void operator() (
    const Titer &it, const cfmav<T0> &in, const vfmav<T0> &out,
    Tstorage &storage, const pocketfft_fftw<T0> &plan, T0 fct, size_t nthreads,
    bool inplace=false) const
    {
    using T = typename Tstorage::datatype;
    if constexpr(is_same<T0, T>::value)
      if (inplace)
        {
        if (in.data()!=out.data())
          copy_input(it, in, out.data()+it.oofs(0));
        plan.exec_copyback(out.data()+it.oofs(0), storage.transformBuf(), fct, forward, nthreads);
        return;
        }
    T *buf1=storage.transformBuf(), *buf2=storage.dataBuf();
    copy_input(it, in, buf2);
    auto res = plan.exec(buf2, buf1, fct, forward, nthreads);
    copy_output(it, res, out);
    }
  template <typename T0, typename Tstorage, typename Titer> DUCC0_NOINLINE void exec_n (
    const Titer &it, const cfmav<T0> &in,
    const vfmav<T0> &out, Tstorage &storage, const pocketfft_fftw<T0> &plan, T0 fct, size_t nvec,
    size_t nthreads) const
    {
    using T = typename Tstorage::datatype;
    size_t dstr = storage.data_stride();
    T *buf1=storage.transformBuf(), *buf2=storage.dataBuf();
    copy_input(it, in, buf2, nvec, dstr);
    for (size_t i=0; i<nvec; ++i)
      plan.exec_copyback(buf2+i*dstr, buf1, fct, forward, nthreads);
    copy_output(it, buf2, out, nvec, dstr);
    }
  template <typename T0> DUCC0_NOINLINE void exec_simple (
    const T0 *in, T0 *out, const pocketfft_fftw<T0> &plan, T0 fct,
    size_t nthreads) const
    {
    if (in!=out) copy_n(in, plan.length(), out);
    plan.exec(out, fct, forward, nthreads);
    }
  };

struct ExecDcst
  {
  bool ortho;
  int type;
  bool cosine;

  template <typename T0, typename Tstorage, typename Tplan, typename Titer>
  DUCC0_NOINLINE void operator() (const Titer &it, const cfmav<T0> &in,
    const vfmav <T0> &out, Tstorage &storage, const Tplan &plan, T0 fct, size_t nthreads,
    bool inplace=false) const
    {
    using T = typename Tstorage::datatype;
    if constexpr(is_same<T0, T>::value)
      if (inplace)
        {
        if (in.data()!=out.data())
          copy_input(it, in, out.data()+it.oofs(0));
        plan.exec_copyback(out.data()+it.oofs(0), storage.transformBuf(), fct, ortho, type, cosine, nthreads);
        return;
        }
    T *buf1=storage.transformBuf(), *buf2=storage.dataBuf();
    copy_input(it, in, buf2);
    auto res = plan.exec(buf2, buf1, fct, ortho, type, cosine, nthreads);
    copy_output(it, res, out);
    }
  template <typename T0, typename Tstorage, typename Tplan, typename Titer> DUCC0_NOINLINE void exec_n (
    const Titer &it, const cfmav<T0> &in,
    const vfmav<T0> &out, Tstorage &storage, const Tplan &plan, T0 fct, size_t nvec,
    size_t nthreads) const
    {
    using T = typename Tstorage::datatype;
    size_t dstr = storage.data_stride();
    T *buf1=storage.transformBuf(), *buf2=storage.dataBuf();
    copy_input(it, in, buf2, nvec, dstr);
    for (size_t i=0; i<nvec; ++i)
      plan.exec_copyback(buf2+i*dstr, buf1, fct, ortho, type, cosine, nthreads);
    copy_output(it, buf2, out, nvec, dstr);
    }
  template <typename T0, typename Tplan> DUCC0_NOINLINE void exec_simple (
    const T0 *in, T0 *out, const Tplan &plan, T0 fct,
    size_t nthreads) const
    {
    if (in!=out) copy_n(in, plan.length(), out);
    plan.exec(out, fct, ortho, type, cosine, nthreads);
    }
  };

template<typename T> DUCC0_NOINLINE void general_r2c(
  const cfmav<T> &in, const vfmav<Cmplx<T>> &out, size_t axis, bool forward, T fct,
  size_t nthreads)
  {
  size_t nth1d = (in.ndim()==1) ? nthreads : 1;
  auto plan = std::make_unique<pocketfft_r<T>>(in.shape(axis));
  size_t len=in.shape(axis);
  execParallel(
    util::thread_count(nthreads, in, axis, fft_simdlen<T>),
    [&](Scheduler &sched) {
    constexpr auto vlen = fft_simdlen<T>;
    TmpStorage<T,T> storage(in.size()/len, len, plan->bufsize(), 1, false);
    multi_iter<vlen> it(in, out, axis, sched.num_threads(), sched.thread_num());
#ifndef DUCC0_NO_SIMD
    if constexpr (vlen>1)
      {
      TmpStorage2<add_vec_t<T, vlen>,T,T> storage2(storage);
      auto dbuf = storage2.dataBuf();
      auto tbuf = storage2.transformBuf();
      while (it.remaining()>=vlen)
        {
        it.advance(vlen);
        copy_input(it, in, dbuf);
        auto res = plan->exec(dbuf, tbuf, fct, true, nth1d);
        auto vout = out.data();
        for (size_t j=0; j<vlen; ++j)
          vout[it.oofs(j,0)].Set(res[0][j]);
        size_t i=1, ii=1;
        if (forward)
          for (; i<len-1; i+=2, ++ii)
            for (size_t j=0; j<vlen; ++j)
              vout[it.oofs(j,ii)].Set(res[i][j], res[i+1][j]);
        else
          for (; i<len-1; i+=2, ++ii)
            for (size_t j=0; j<vlen; ++j)
              vout[it.oofs(j,ii)].Set(res[i][j], -res[i+1][j]);
        if (i<len)
          for (size_t j=0; j<vlen; ++j)
            vout[it.oofs(j,ii)].Set(res[i][j]);
        }
      }
    if constexpr (vlen>2)
      if constexpr (simd_exists<T,vlen/2>)
        if (it.remaining()>=vlen/2)
          {
          TmpStorage2<add_vec_t<T, vlen/2>,T,T> storage2(storage);
          auto dbuf = storage2.dataBuf();
          auto tbuf = storage2.transformBuf();
          it.advance(vlen/2);
          copy_input(it, in, dbuf);
          auto res = plan->exec(dbuf, tbuf, fct, true, nth1d);
          auto vout = out.data();
          for (size_t j=0; j<vlen/2; ++j)
            vout[it.oofs(j,0)].Set(res[0][j]);
          size_t i=1, ii=1;
          if (forward)
            for (; i<len-1; i+=2, ++ii)
              for (size_t j=0; j<vlen/2; ++j)
                vout[it.oofs(j,ii)].Set(res[i][j], res[i+1][j]);
          else
            for (; i<len-1; i+=2, ++ii)
              for (size_t j=0; j<vlen/2; ++j)
                vout[it.oofs(j,ii)].Set(res[i][j], -res[i+1][j]);
          if (i<len)
            for (size_t j=0; j<vlen/2; ++j)
              vout[it.oofs(j,ii)].Set(res[i][j]);
          }
    if constexpr (vlen>4)
      if constexpr( simd_exists<T,vlen/4>)
        if (it.remaining()>=vlen/4)
          {
          TmpStorage2<add_vec_t<T, vlen/4>,T,T> storage2(storage);
          auto dbuf = storage2.dataBuf();
          auto tbuf = storage2.transformBuf();
          it.advance(vlen/4);
          copy_input(it, in, dbuf);
          auto res = plan->exec(dbuf, tbuf, fct, true, nth1d);
          auto vout = out.data();
          for (size_t j=0; j<vlen/4; ++j)
            vout[it.oofs(j,0)].Set(res[0][j]);
          size_t i=1, ii=1;
          if (forward)
            for (; i<len-1; i+=2, ++ii)
              for (size_t j=0; j<vlen/4; ++j)
                vout[it.oofs(j,ii)].Set(res[i][j], res[i+1][j]);
          else
            for (; i<len-1; i+=2, ++ii)
              for (size_t j=0; j<vlen/4; ++j)
                vout[it.oofs(j,ii)].Set(res[i][j], -res[i+1][j]);
          if (i<len)
            for (size_t j=0; j<vlen/4; ++j)
              vout[it.oofs(j,ii)].Set(res[i][j]);
          }
#endif
    {
    TmpStorage2<T,T,T> storage2(storage);
    auto dbuf = storage2.dataBuf();
    auto tbuf = storage2.transformBuf();
    while (it.remaining()>0)
      {
      it.advance(1);
      copy_input(it, in, dbuf);
      auto res = plan->exec(dbuf, tbuf, fct, true, nth1d);
      auto vout = out.data();
      vout[it.oofs(0)].Set(res[0]);
      size_t i=1, ii=1;
      if (forward)
        for (; i<len-1; i+=2, ++ii)
          vout[it.oofs(ii)].Set(res[i], res[i+1]);
      else
        for (; i<len-1; i+=2, ++ii)
          vout[it.oofs(ii)].Set(res[i], -res[i+1]);
      if (i<len)
        vout[it.oofs(ii)].Set(res[i]);
      }
    }
    });  // end of parallel region
  }
template<typename T> DUCC0_NOINLINE void general_c2r(
  const cfmav<Cmplx<T>> &in, const vfmav<T> &out, size_t axis, bool forward, T fct,
  size_t nthreads)
  {
  size_t nth1d = (in.ndim()==1) ? nthreads : 1;
  auto plan = std::make_unique<pocketfft_r<T>>(out.shape(axis));
  size_t len=out.shape(axis);
  execParallel(
    util::thread_count(nthreads, in, axis, fft_simdlen<T>),
    [&](Scheduler &sched) {
      constexpr auto vlen = fft_simdlen<T>;
      TmpStorage<T,T> storage(out.size()/len, len, plan->bufsize(), 1, false);
      multi_iter<vlen> it(in, out, axis, sched.num_threads(), sched.thread_num());
#ifndef DUCC0_NO_SIMD
      if constexpr (vlen>1)
        {
        TmpStorage2<add_vec_t<T, vlen>,T,T> storage2(storage);
        auto dbuf = storage2.dataBuf();
        auto tbuf = storage2.transformBuf();
        while (it.remaining()>=vlen)
          {
          it.advance(vlen);
          for (size_t j=0; j<vlen; ++j)
            dbuf[0][j]=in.raw(it.iofs(j,0)).r;
          {
          size_t i=1, ii=1;
          if (forward)
            for (; i<len-1; i+=2, ++ii)
              for (size_t j=0; j<vlen; ++j)
                {
                dbuf[i  ][j] =  in.raw(it.iofs(j,ii)).r;
                dbuf[i+1][j] = -in.raw(it.iofs(j,ii)).i;
                }
          else
            for (; i<len-1; i+=2, ++ii)
              for (size_t j=0; j<vlen; ++j)
                {
                dbuf[i  ][j] = in.raw(it.iofs(j,ii)).r;
                dbuf[i+1][j] = in.raw(it.iofs(j,ii)).i;
                }
          if (i<len)
            for (size_t j=0; j<vlen; ++j)
              dbuf[i][j] = in.raw(it.iofs(j,ii)).r;
          }
          auto res = plan->exec(dbuf, tbuf, fct, false, nth1d);
          copy_output(it, res, out);
          }
        }
      if constexpr (vlen>2)
        if constexpr (simd_exists<T,vlen/2>)
          if (it.remaining()>=vlen/2)
            {
            TmpStorage2<add_vec_t<T, vlen/2>,T,T> storage2(storage);
            auto dbuf = storage2.dataBuf();
            auto tbuf = storage2.transformBuf();
            it.advance(vlen/2);
            for (size_t j=0; j<vlen/2; ++j)
              dbuf[0][j]=in.raw(it.iofs(j,0)).r;
            {
            size_t i=1, ii=1;
            if (forward)
              for (; i<len-1; i+=2, ++ii)
                for (size_t j=0; j<vlen/2; ++j)
                  {
                  dbuf[i  ][j] =  in.raw(it.iofs(j,ii)).r;
                  dbuf[i+1][j] = -in.raw(it.iofs(j,ii)).i;
                  }
            else
              for (; i<len-1; i+=2, ++ii)
                for (size_t j=0; j<vlen/2; ++j)
                  {
                  dbuf[i  ][j] = in.raw(it.iofs(j,ii)).r;
                  dbuf[i+1][j] = in.raw(it.iofs(j,ii)).i;
                  }
            if (i<len)
              for (size_t j=0; j<vlen/2; ++j)
                dbuf[i][j] = in.raw(it.iofs(j,ii)).r;
            }
            auto res = plan->exec(dbuf, tbuf, fct, false, nth1d);
            copy_output(it, res, out);
            }
      if constexpr (vlen>4)
        if constexpr(simd_exists<T,vlen/4>)
          if (it.remaining()>=vlen/4)
            {
            TmpStorage2<add_vec_t<T, vlen/4>,T,T> storage2(storage);
            auto dbuf = storage2.dataBuf();
            auto tbuf = storage2.transformBuf();
            it.advance(vlen/4);
            for (size_t j=0; j<vlen/4; ++j)
              dbuf[0][j]=in.raw(it.iofs(j,0)).r;
            {
            size_t i=1, ii=1;
            if (forward)
              for (; i<len-1; i+=2, ++ii)
                for (size_t j=0; j<vlen/4; ++j)
                  {
                  dbuf[i  ][j] =  in.raw(it.iofs(j,ii)).r;
                  dbuf[i+1][j] = -in.raw(it.iofs(j,ii)).i;
                  }
            else
              for (; i<len-1; i+=2, ++ii)
                for (size_t j=0; j<vlen/4; ++j)
                  {
                  dbuf[i  ][j] = in.raw(it.iofs(j,ii)).r;
                  dbuf[i+1][j] = in.raw(it.iofs(j,ii)).i;
                  }
            if (i<len)
              for (size_t j=0; j<vlen/4; ++j)
                dbuf[i][j] = in.raw(it.iofs(j,ii)).r;
            }
            auto res = plan->exec(dbuf, tbuf, fct, false, nth1d);
            copy_output(it, res, out);
            }
#endif
      {
      TmpStorage2<T,T,T> storage2(storage);
      auto dbuf = storage2.dataBuf();
      auto tbuf = storage2.transformBuf();
      while (it.remaining()>0)
        {
        it.advance(1);
        dbuf[0]=in.raw(it.iofs(0)).r;
        {
        size_t i=1, ii=1;
        if (forward)
          for (; i<len-1; i+=2, ++ii)
            {
            dbuf[i  ] =  in.raw(it.iofs(ii)).r;
            dbuf[i+1] = -in.raw(it.iofs(ii)).i;
            }
        else
          for (; i<len-1; i+=2, ++ii)
            {
            dbuf[i  ] = in.raw(it.iofs(ii)).r;
            dbuf[i+1] = in.raw(it.iofs(ii)).i;
            }
        if (i<len)
          dbuf[i] = in.raw(it.iofs(ii)).r;
        }
        auto res = plan->exec(dbuf, tbuf, fct, false, nth1d);
        copy_output(it, res, out);
        }
      }
    });  // end of parallel region
  }

struct ExecR2R
  {
  bool r2c, forward;

  template <typename T0, typename Tstorage, typename Titer> DUCC0_NOINLINE void operator() (
    const Titer &it, const cfmav<T0> &in, const vfmav<T0> &out, Tstorage &storage,
    const pocketfft_r<T0> &plan, T0 fct, size_t nthreads,
    bool inplace=false) const
    {
    using T = typename Tstorage::datatype;
    if constexpr(is_same<T0, T>::value)
      if (inplace)
        {
        T *buf1=storage.transformBuf(), *buf2=out.data()+it.oofs(0);
        if (in.data()!=buf2)
          copy_input(it, in, buf2);
        if ((!r2c) && forward)
          for (size_t i=2; i<it.length_out(); i+=2)
            buf2[i] = -buf2[i];
        plan.exec_copyback(buf2, buf1, fct, r2c, nthreads);
        if (r2c && (!forward))
          for (size_t i=2; i<it.length_out(); i+=2)
            buf2[i] = -buf2[i];
        return;
        }

    T *buf1=storage.transformBuf(), *buf2=storage.dataBuf();
    copy_input(it, in, buf2);
    if ((!r2c) && forward)
      for (size_t i=2; i<it.length_out(); i+=2)
        buf2[i] = -buf2[i];
    auto res = plan.exec(buf2, buf1, fct, r2c, nthreads);
    if (r2c && (!forward))
      for (size_t i=2; i<it.length_out(); i+=2)
        res[i] = -res[i];
    copy_output(it, res, out);
    }
  template <typename T0, typename Tstorage, typename Titer> DUCC0_NOINLINE void exec_n (
    const Titer &it, const cfmav<T0> &in,
    const vfmav<T0> &out, Tstorage &storage, const pocketfft_r<T0> &plan, T0 fct, size_t nvec,
    size_t nthreads) const
    {
    using T = typename Tstorage::datatype;
    size_t dstr = storage.data_stride();
    T *buf1=storage.transformBuf(), *buf2=storage.dataBuf();
    copy_input(it, in, buf2, nvec, dstr);
    if ((!r2c) && forward)
      for (size_t k=0; k<nvec; ++k)
        for (size_t i=2; i<it.length_out(); i+=2)
          buf2[i+k*dstr] = -buf2[i+k*dstr];
    for (size_t i=0; i<nvec; ++i)
      plan.exec_copyback(buf2+i*dstr, buf1, fct, r2c, nthreads);
    if (r2c && (!forward))
      for (size_t k=0; k<nvec; ++k)
        for (size_t i=2; i<it.length_out(); i+=2)
          buf2[i+k*dstr] = -buf2[i+k*dstr];
    copy_output(it, buf2, out, nvec, dstr);
    }
  template <typename T0> DUCC0_NOINLINE void exec_simple (
    const T0 *in, T0 *out, const pocketfft_r<T0> &plan, T0 fct,
    size_t nthreads) const
    {
    if (in!=out) copy_n(in, plan.length(), out);
    if ((!r2c) && forward)
      for (size_t i=2; i<plan.length(); i+=2)
        out[i] = -out[i];
    plan.exec(out, fct, r2c, nthreads);
    if (r2c && (!forward))
      for (size_t i=2; i<plan.length(); i+=2)
        out[i] = -out[i];
    }
  };

template<typename T> class Long1dPlan: public UnityRoots<T,complex<T>>
  {
  public:
    Long1dPlan(size_t length, bool)
      : UnityRoots<T,complex<T>>(length) {}
  };

template<typename T> DUCC0_NOINLINE void c2c(const cfmav<std::complex<T>> &in,
  const vfmav<std::complex<T>> &out, const shape_t &axes, bool forward,
  T fct, size_t nthreads)
  {
  util::sanity_check_onetype(in, out, in.data()==out.data(), axes);
  if (in.size()==0) return;

  // special treatment for long 1D transforms (Bailey's algorithm)
  // TODO:
  //  - if not in-place and out has no critical stride, the "tmp" array can be avoided
  if ((in.ndim()==1) && (in.shape(0)>=65536*2))
    {
    size_t ip = in.shape(0);
    auto factors = util1d::prime_factors(ip);
    sort(factors.begin(), factors.end(), std::greater<size_t>());
    size_t f1=1, f2=1;
    for (auto fct: factors)
      (f2>f1) ? f1*=fct : f2*=fct;
    if (f1>f2) swap(f1,f2);
    if (f1>=16) // 2D algorithm makes sense
      {
      auto istr=in.stride(0);
      auto ostr=out.stride(0);
      cmav<std::complex<T>,2> in2 (in.data(), {f1,f2}, {ptrdiff_t(f2)*istr, istr});
      auto tmp (vmav<std::complex<T>,2>::build_noncritical({f1,f2}));
      vmav<std::complex<T>,2> out2 (out.data(), {f1,f2}, {ostr, ptrdiff_t(f1)*ostr});
      auto fin2(in2.to_fmav());
      auto ftmp(tmp.to_fmav());
      auto fout2(out2.to_fmav());
      c2c(fin2, ftmp, {0}, forward, T(1), nthreads);
      auto roots_p = get_plan<Long1dPlan<T>>(ip);
      const auto &roots(*roots_p);
      if (forward)
        execStatic(f1, nthreads, 0, [&](Scheduler &sched) {
          while (auto rng=sched.getNext())
            for (auto i=rng.lo; i<rng.hi; ++i)
              for (size_t j=0; j<f2; ++j)
                tmp(i,j) *= conj(roots[i*j]);
          });
      else
        execStatic(f1, nthreads, 0, [&](Scheduler &sched) {
          while (auto rng=sched.getNext())
            for (auto i=rng.lo; i<rng.hi; ++i)
              for (size_t j=0; j<f2; ++j)
                tmp(i,j) *= roots[i*j];
          });
      c2c(ftmp, fout2, {1}, forward, fct, nthreads);
      return;
      }
    }
 
  const auto &in2(reinterpret_cast<const cfmav<Cmplx<T> >&>(in));
  const auto &out2(reinterpret_cast<const vfmav<Cmplx<T> >&>(out));
  if ((axes.size()>1) && (in.data()!=out.data())) // optimize axis order
    {
    if ((in.stride(axes[0])!=1)&&(out.stride(axes[0])==1))
      {
      shape_t axes2(axes);
      swap(axes2[0],axes2.back());
      general_nd<pocketfft_c<T>>(in2, out2, axes2, fct, nthreads, ExecC2C{forward});
      return;
      }
    for (size_t i=1; i<axes.size(); ++i)
      if (in.stride(axes[i])==1)
        {
        shape_t axes2(axes);
        swap(axes2[0],axes2[i]);
        general_nd<pocketfft_c<T>>(in2, out2, axes2, fct, nthreads, ExecC2C{forward});
        return;
        }
    }
  general_nd<pocketfft_c<T>>(in2, out2, axes, fct, nthreads, ExecC2C{forward});
  }

template<typename T> DUCC0_NOINLINE void dct(const cfmav<T> &in, const vfmav<T> &out,
  const shape_t &axes, int type, T fct, bool ortho, size_t nthreads)
  {
  if ((type<1) || (type>4)) throw std::invalid_argument("invalid DCT type");
  util::sanity_check_onetype(in, out, in.data()==out.data(), axes);
  if (in.size()==0) return;
  const ExecDcst exec{ortho, type, true};
  if (type==1)
    general_nd<T_dct1<T>>(in, out, axes, fct, nthreads, exec);
  else if (type==4)
    general_nd<T_dcst4<T>>(in, out, axes, fct, nthreads, exec);
  else
    general_nd<T_dcst23<T>>(in, out, axes, fct, nthreads, exec);
  }

template<typename T> DUCC0_NOINLINE void dst(const cfmav<T> &in, const vfmav<T> &out,
  const shape_t &axes, int type, T fct, bool ortho, size_t nthreads)
  {
  if ((type<1) || (type>4)) throw std::invalid_argument("invalid DST type");
  util::sanity_check_onetype(in, out, in.data()==out.data(), axes);
  if (in.size()==0) return;
  const ExecDcst exec{ortho, type, false};
  if (type==1)
    general_nd<T_dst1<T>>(in, out, axes, fct, nthreads, exec);
  else if (type==4)
    general_nd<T_dcst4<T>>(in, out, axes, fct, nthreads, exec);
  else
    general_nd<T_dcst23<T>>(in, out, axes, fct, nthreads, exec);
  }

template<typename T> DUCC0_NOINLINE void r2c(const cfmav<T> &in,
  const vfmav<std::complex<T>> &out, size_t axis, bool forward, T fct,
  size_t nthreads)
  {
  util::sanity_check_cr(out, in, axis);
  if (in.size()==0) return;
  const auto &out2(reinterpret_cast<const vfmav<Cmplx<T>>&>(out));
  general_r2c(in, out2, axis, forward, fct, nthreads);
  }

template<typename T> DUCC0_NOINLINE void r2c(const cfmav<T> &in,
  const vfmav<std::complex<T>> &out, const shape_t &axes,
  bool forward, T fct, size_t nthreads)
  {
  util::sanity_check_cr(out, in, axes);
  if (in.size()==0) return;
  r2c(in, out, axes.back(), forward, fct, nthreads);
  if (axes.size()==1) return;

  auto newaxes = shape_t{axes.begin(), --axes.end()};
  c2c(out, out, newaxes, forward, T(1), nthreads);
  }

template<typename T> DUCC0_NOINLINE void c2r(const cfmav<std::complex<T>> &in,
  const vfmav<T> &out,  size_t axis, bool forward, T fct, size_t nthreads)
  {
  util::sanity_check_cr(in, out, axis);
  if (in.size()==0) return;
  const auto &in2(reinterpret_cast<const cfmav<Cmplx<T>>&>(in));
  general_c2r(in2, out, axis, forward, fct, nthreads);
  }

template<typename T> DUCC0_NOINLINE void c2r(const cfmav<std::complex<T>> &in,
  const vfmav<T> &out, const shape_t &axes, bool forward, T fct,
  size_t nthreads)
  {
  if (axes.size()==1)
    return c2r(in, out, axes[0], forward, fct, nthreads);
  util::sanity_check_cr(in, out, axes);
  if (in.size()==0) return;
  auto atmp(vfmav<std::complex<T>>::build_noncritical(in.shape(), UNINITIALIZED));
  auto newaxes = shape_t{axes.begin(), --axes.end()};
  c2c(in, atmp, newaxes, forward, T(1), nthreads);
  c2r(atmp, out, axes.back(), forward, fct, nthreads);
  }

template<typename T> DUCC0_NOINLINE void c2r_mut(const vfmav<std::complex<T>> &in,
  const vfmav<T> &out, const shape_t &axes, bool forward, T fct,
  size_t nthreads)
  {
  if (axes.size()==1)
    return c2r(in, out, axes[0], forward, fct, nthreads);
  util::sanity_check_cr(in, out, axes);
  if (in.size()==0) return;
  auto newaxes = shape_t{axes.begin(), --axes.end()};
  c2c(in, in, newaxes, forward, T(1), nthreads);
  c2r(in, out, axes.back(), forward, fct, nthreads);
  }

template<typename T> DUCC0_NOINLINE void r2r_fftpack(const cfmav<T> &in,
  const vfmav<T> &out, const shape_t &axes, bool real2hermitian, bool forward,
  T fct, size_t nthreads)
  {
  util::sanity_check_onetype(in, out, in.data()==out.data(), axes);
  if (in.size()==0) return;
  general_nd<pocketfft_r<T>>(in, out, axes, fct, nthreads,
    ExecR2R{real2hermitian, forward});
  }

template<typename T> DUCC0_NOINLINE void r2r_fftw(const cfmav<T> &in,
  const vfmav<T> &out, const shape_t &axes, bool forward,
  T fct, size_t nthreads)
  {
  util::sanity_check_onetype(in, out, in.data()==out.data(), axes);
  if (in.size()==0) return;
  general_nd<pocketfft_fftw<T>>(in, out, axes, fct, nthreads,
    ExecFFTW{forward});
  }

template<typename T> DUCC0_NOINLINE void r2r_separable_hartley(const cfmav<T> &in,
  const vfmav<T> &out, const shape_t &axes, T fct, size_t nthreads)
  {
  util::sanity_check_onetype(in, out, in.data()==out.data(), axes);
  if (in.size()==0) return;
  general_nd<pocketfft_hartley<T>>(in, out, axes, fct, nthreads,
    ExecHartley{}, false);
  }

template<typename T> DUCC0_NOINLINE void r2r_separable_fht(const cfmav<T> &in,
  const vfmav<T> &out, const shape_t &axes, T fct, size_t nthreads)
  {
  util::sanity_check_onetype(in, out, in.data()==out.data(), axes);
  if (in.size()==0) return;
  general_nd<pocketfft_fht<T>>(in, out, axes, fct, nthreads,
    ExecFHT{}, false);
  }

template<typename T0, typename T1, typename Func> void hermiteHelper(size_t idim, ptrdiff_t iin,
  ptrdiff_t iout0, ptrdiff_t iout1, const cfmav<T0> &c,
  const vfmav<T1> &r, const shape_t &axes, Func func, size_t nthreads)
  {
  auto cstr=c.stride(idim), str=r.stride(idim);
  auto len=r.shape(idim);

  if (idim+1==c.ndim())  // last dimension, not much gain in parallelizing
    {
    if (idim==axes.back())  // halfcomplex axis
      for (size_t i=0,ic=0; i<len/2+1; ++i,ic=len-i)
        func (c.raw(iin+i*cstr), r.raw(iout0+i*str), r.raw(iout1+ic*str));
    else if (find(axes.begin(), axes.end(), idim) != axes.end())  // FFT axis
      for (size_t i=0,ic=0; i<len; ++i,ic=len-i)
        func (c.raw(iin+i*cstr), r.raw(iout0+i*str), r.raw(iout1+ic*str));
    else  // non-FFT axis
      for (size_t i=0; i<len; ++i)
        func (c.raw(iin+i*cstr), r.raw(iout0+i*str), r.raw(iout1+i*str));
    }
  else
    {
    if (idim==axes.back())
      {
      if (nthreads==1)
        for (size_t i=0,ic=0; i<len/2+1; ++i,ic=len-i)
          hermiteHelper(idim+1, iin+i*cstr, iout0+i*str, iout1+ic*str, c, r, axes, func, 1);
      else
        execParallel(0, len/2+1, nthreads, [&](size_t lo, size_t hi)
          {
          for (size_t i=lo,ic=(i==0?0:len-i); i<hi; ++i,ic=len-i)
            hermiteHelper(idim+1, iin+i*cstr, iout0+i*str, iout1+ic*str, c, r, axes, func, 1);
          });
      }
    else if (find(axes.begin(), axes.end(), idim) != axes.end())
      {
      if (nthreads==1)
        for (size_t i=0,ic=0; i<len; ++i,ic=len-i)
          hermiteHelper(idim+1, iin+i*cstr, iout0+i*str, iout1+ic*str, c, r, axes, func, 1);
      else
        execParallel(0, len/2+1, nthreads, [&](size_t lo, size_t hi)
          {
          for (size_t i=lo,ic=(i==0?0:len-i); i<hi; ++i,ic=len-i)
            {
            size_t io0=iout0+i*str, io1=iout1+ic*str;
            hermiteHelper(idim+1, iin+i*cstr, io0, io1, c, r, axes, func, 1);
            if (i!=ic)
              hermiteHelper(idim+1, iin+ic*cstr, io1, io0, c, r, axes, func, 1);
            }
          });
      }
    else
      {
      if (nthreads==1)
        for (size_t i=0; i<len; ++i)
          hermiteHelper(idim+1, iin+i*cstr, iout0+i*str, iout1+i*str, c, r, axes, func, 1);
      else
        execParallel(0, len, nthreads, [&](size_t lo, size_t hi)
          {
          for (size_t i=lo; i<hi; ++i)
            hermiteHelper(idim+1, iin+i*cstr, iout0+i*str, iout1+i*str, c, r, axes, func, 1);
          });
      }
    }
  }

template<typename T> void oscarize(const vfmav<T> &data, size_t ax0, size_t ax1,
  size_t nthreads)
  {
  auto nu=data.shape(ax0), nv=data.shape(ax1);
  if ((nu<3)||(nv<3)) return;
  vector<slice> slc(data.ndim());
  slc[ax0] = slice(1,(nu+1)/2);
  slc[ax1] = slice(1,(nv+1)/2);
  auto all = subarray(data, slc);
  slc[ax0] = slice(nu-1,nu/2,-1);
  auto ahl = subarray(data, slc);
  slc[ax1] = slice(nv-1,nv/2,-1);
  auto ahh = subarray(data, slc);
  slc[ax0] = slice(1,(nu+1)/2);
  auto alh = subarray(data, slc);
  mav_apply([](T &ll, T &hl, T &hh, T &lh)
    {
    T tll=ll, thl=hl, tlh=lh, thh=hh;
    T v = T(0.5)*(tll+tlh+thl+thh);
    ll = v-thh;
    hl = v-tlh;
    lh = v-thl;
    hh = v-tll;
    }, nthreads, all, ahl, ahh, alh);
  }

template<typename T> void r2r_genuine_hartley(const cfmav<T> &in,
  const vfmav<T> &out, const shape_t &axes, T fct, size_t nthreads)
  {
  if (axes.size()==1)
    return r2r_separable_hartley(in, out, axes, fct, nthreads);
  if (axes.size()==2)
    {
    r2r_separable_hartley(in, out, axes, fct, nthreads);
    oscarize(out, axes[0], axes[1], nthreads);
    return;
    }
  util::sanity_check_onetype(in, out, in.data()==out.data(), axes);
  if (in.size()==0) return;
  shape_t tshp(in.shape());
  tshp[axes.back()] = tshp[axes.back()]/2+1;
  auto atmp(vfmav<std::complex<T>>::build_noncritical(tshp, UNINITIALIZED));
  r2c(in, atmp, axes, true, fct, nthreads);
  hermiteHelper(0, 0, 0, 0, atmp, out, axes, [](const std::complex<T> &c, T &r0, T &r1)
    {
    auto ccopy = c;
    r0 = ccopy.real()+ccopy.imag();
    r1 = ccopy.real()-ccopy.imag();
    }, nthreads);
  }

template<typename T> void r2r_genuine_fht(const cfmav<T> &in,
  const vfmav<T> &out, const shape_t &axes, T fct, size_t nthreads)
  {
  if (axes.size()==1)
    return r2r_separable_fht(in, out, axes, fct, nthreads);
  if (axes.size()==2)
    {
    r2r_separable_fht(in, out, axes, fct, nthreads);
    oscarize(out, axes[0], axes[1], nthreads);
    return;
    }
  util::sanity_check_onetype(in, out, in.data()==out.data(), axes);
  if (in.size()==0) return;
  shape_t tshp(in.shape());
  tshp[axes.back()] = tshp[axes.back()]/2+1;
  auto atmp(vfmav<std::complex<T>>::build_noncritical(tshp, UNINITIALIZED));
  r2c(in, atmp, axes, true, fct, nthreads);
  hermiteHelper(0, 0, 0, 0, atmp, out, axes, [](const std::complex<T> &c, T &r0, T &r1)
    {
    auto ccopy = c;
    r0 = ccopy.real()-ccopy.imag();
    r1 = ccopy.real()+ccopy.imag();
    }, nthreads);
  }

template<typename Tplan, typename T0, typename T, typename Exec>
DUCC0_NOINLINE void general_convolve_axis(const cfmav<T> &in, const vfmav<T> &out,
  const size_t axis, const cmav<T,1> &kernel, size_t nthreads,
  const Exec &exec)
  {
  std::unique_ptr<Tplan> plan1, plan2;

  size_t l_in=in.shape(axis), l_out=out.shape(axis);
  MR_assert(kernel.size()==l_in, "bad kernel size");
  plan1 = std::make_unique<Tplan>(l_in);
  plan2 = std::make_unique<Tplan>(l_out);
  size_t bufsz = max(plan1->bufsize(), plan2->bufsize());

  vmav<T,1> fkernel({kernel.shape(0)}, UNINITIALIZED);
  for (size_t i=0; i<kernel.shape(0); ++i)
    fkernel(i) = kernel(i);
  plan1->exec(fkernel.data(), T0(1)/T0(l_in), true, nthreads);

  execParallel(
    util::thread_count(nthreads, in, axis, fft_simdlen<T0>),
    [&](Scheduler &sched) {
      constexpr auto vlen = fft_simdlen<T0>;
      TmpStorage<T,T0> storage(in.size()/l_in, l_in+l_out, bufsz, 1, false);
      multi_iter<vlen> it(in, out, axis, sched.num_threads(), sched.thread_num());
#ifndef DUCC0_NO_SIMD
      if constexpr (vlen>1)
        {
        TmpStorage2<add_vec_t<T, vlen>,T,T0> storage2(storage);
        while (it.remaining()>=vlen)
          {
          it.advance(vlen);
          exec(it, in, out, storage2, *plan1, *plan2, fkernel);
          }
        }
      if constexpr (vlen>2)
        if constexpr (simd_exists<T,vlen/2>)
          if (it.remaining()>=vlen/2)
            {
            TmpStorage2<add_vec_t<T, vlen/2>,T,T0> storage2(storage);
            it.advance(vlen/2);
            exec(it, in, out, storage2, *plan1, *plan2, fkernel);
            }
      if constexpr (vlen>4)
        if constexpr (simd_exists<T,vlen/4>)
          if (it.remaining()>=vlen/4)
            {
            TmpStorage2<add_vec_t<T, vlen/4>,T,T0> storage2(storage);
            it.advance(vlen/4);
            exec(it, in, out, storage2, *plan1, *plan2, fkernel);
            }
#endif
      {
      TmpStorage2<T,T,T0> storage2(storage);
      while (it.remaining()>0)
        {
        it.advance(1);
        exec(it, in, out, storage2, *plan1, *plan2, fkernel);
        }
      }
    });  // end of parallel region
  }

struct ExecConv1R
  {
  template <typename T0, typename Tstorage, typename Titer> void operator() (
    const Titer &it, const cfmav<T0> &in, const vfmav<T0> &out,
    Tstorage &storage, const pocketfft_r<T0> &plan1, const pocketfft_r<T0> &plan2,
    const cmav<T0,1> &fkernel) const
    {
    using T = typename Tstorage::datatype;
    size_t l_in = plan1.length(),
           l_out = plan2.length(),
           l_min = std::min(l_in, l_out);
    T *buf1=storage.transformBuf(), *buf2=storage.dataBuf();
    copy_input(it, in, buf2);
    plan1.exec_copyback(buf2, buf1, T0(1), true);
    auto res = buf2;
    {
    res[0] *= fkernel(0);
    size_t i;
    for (i=1; 2*i<l_min; ++i)
      {
      Cmplx<T> t1(res[2*i-1], res[2*i]);
      Cmplx<T0> t2(fkernel(2*i-1), fkernel(2*i));
      auto t3 = t1*t2;
      res[2*i-1] = t3.r;
      res[2*i] = t3.i;
      }
    if (2*i==l_min)
      {
      if (l_min<l_out) // padding
        res[2*i-1] *= fkernel(2*i-1)*T0(0.5);
      else if (l_min<l_in) // truncation
        {
        Cmplx<T> t1(res[2*i-1], res[2*i]);
        Cmplx<T0> t2(fkernel(2*i-1), fkernel(2*i));
        res[2*i-1] = (t1*t2).r*T0(2);
        }
      else
        res[2*i-1] *= fkernel(2*i-1);
      }
    }
    for (size_t i=l_in; i<l_out; ++i) res[i] = T(0);
    res = plan2.exec(res, buf1, T0(1), false);
    copy_output(it, res, out);
    }
  };
struct ExecConv1C
  {
  template <typename T0, typename Tstorage, typename Titer> void operator() (
    const Titer &it, const cfmav<Cmplx<T0>> &in, const vfmav<Cmplx<T0>> &out,
    Tstorage &storage, const pocketfft_c<T0> &plan1, const pocketfft_c<T0> &plan2,
    const cmav<Cmplx<T0>,1> &fkernel) const
    {
    using T = typename Tstorage::datatype;
    size_t l_in = plan1.length(),
           l_out = plan2.length(),
           l_min = std::min(l_in, l_out);
    T *buf1=storage.transformBuf(), *buf2=storage.dataBuf();
    copy_input(it, in, buf2);
    auto res = plan1.exec(buf2, buf1, T0(1), true);
    auto res2 = buf2+l_in;
    {
    res2[0] = res[0]*fkernel(0);
    size_t i;
    for (i=1; 2*i<l_min; ++i)
      {
      res2[i] = res[i]*fkernel(i);
      res2[l_out-i] = res[l_in-i]*fkernel(l_in-i);
      }
    if (2*i==l_min)
      {
      if (l_min<l_out) // padding
        res2[l_out-i] = res2[i] = res[i]*fkernel(i)*T0(.5);
      else if (l_min<l_in) // truncation
        res2[i] = res[i]*fkernel(i) + res[l_in-i]*fkernel(l_in-i);
      else
        res2[i] = res[i]*fkernel(i);
      ++i;
      }
    for (; 2*i<=l_out; ++i)
      res2[i] = res2[l_out-i] = T(0,0);
    }
    res = plan2.exec(res2, buf1, T0(1), false);
    copy_output(it, res, out);
    }
  };

template<typename T> DUCC0_NOINLINE void convolve_axis(const cfmav<T> &in,
  const vfmav<T> &out, size_t axis, const cmav<T,1> &kernel, size_t nthreads)
  {
  MR_assert(axis<in.ndim(), "bad axis number");
  MR_assert(in.ndim()==out.ndim(), "dimensionality mismatch");
  if (in.data()==out.data())
    MR_assert(in.stride()==out.stride(), "strides mismatch");
  for (size_t i=0; i<in.ndim(); ++i)
    if (i!=axis)
      MR_assert(in.shape(i)==out.shape(i), "shape mismatch");
  if (in.size()==0) return;
  general_convolve_axis<pocketfft_r<T>, T>(in, out, axis, kernel, nthreads,
    ExecConv1R());
  }
template<typename T> DUCC0_NOINLINE void convolve_axis(const cfmav<complex<T>> &in,
  const vfmav<complex<T>> &out, size_t axis, const cmav<complex<T>,1> &kernel,
  size_t nthreads)
  {
  MR_assert(axis<in.ndim(), "bad axis number");
  MR_assert(in.ndim()==out.ndim(), "dimensionality mismatch");
  if (in.data()==out.data())
    MR_assert(in.stride()==out.stride(), "strides mismatch");
  for (size_t i=0; i<in.ndim(); ++i)
    if (i!=axis)
      MR_assert(in.shape(i)==out.shape(i), "shape mismatch");
  if (in.size()==0) return;
  const auto &in2(reinterpret_cast<const cfmav<Cmplx<T>>&>(in));
  const auto &out2(reinterpret_cast<const vfmav<Cmplx<T>>&>(out));
  const auto &kernel2(reinterpret_cast<const cmav<Cmplx<T>,1>&>(kernel));
  general_convolve_axis<pocketfft_c<T>, T>(in2, out2, axis, kernel2, nthreads,
    ExecConv1C());
  }

} // namespace detail_fft

} // namespace ducc0

#endif // POCKETFFT_HDRONLY_H
