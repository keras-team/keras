/*
This file is part of the ducc FFT library

Copyright (C) 2010-2023 Max-Planck-Society
Copyright (C) 2019 Peter Bell

For the odd-sized DCT-IV transforms:
  Copyright (C) 2003, 2007-14 Matteo Frigo
  Copyright (C) 2003, 2007-14 Massachusetts Institute of Technology

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

#ifndef DUCC0_FFT1D_IMPL_H
#define DUCC0_FFT1D_IMPL_H

#include <memory>
#include <cstddef>
#include <algorithm>
#include <functional>
#include <type_traits>
#include <utility>
#include <vector>
#include <typeinfo>
#include <typeindex>
#include "ducc0/infra/useful_macros.h"
#include "ducc0/math/cmplx.h"
#include "ducc0/infra/error_handling.h"
#include "ducc0/infra/aligned_array.h"
#include "ducc0/infra/simd.h"
#include "ducc0/infra/threading.h"
#include "ducc0/math/unity_roots.h"
#include "ducc0/fft/fft.h"

namespace ducc0 {

namespace detail_fft {

using namespace std;

// the next line is necessary to address some sloppy name choices in AdaptiveCpp
using std::min, std::max;

template<typename T> constexpr inline size_t fft1d_simdlen
  = min<size_t>(8, native_simd<T>::size());
template<> constexpr inline size_t fft1d_simdlen<double>
  = min<size_t>(4, native_simd<double>::size());
template<> constexpr inline size_t fft1d_simdlen<float>
  = min<size_t>(8, native_simd<float>::size());
template<typename T> using fft1d_simd = typename simd_select<T,fft1d_simdlen<T>>::type;
template<typename T> constexpr inline bool fft1d_simd_exists = (fft1d_simdlen<T> > 1);

// Always use std:: for <cmath> functions
template <typename T> T cos(T) = delete;
template <typename T> T sin(T) = delete;
template <typename T> T sqrt(T) = delete;

template<bool fwd, typename T> void ROTX90(Cmplx<T> &a)
  { auto tmp_= fwd ? -a.r : a.r; a.r = fwd ? a.i : -a.i; a.i=tmp_; }

#define POCKETFFT_EXEC_DISPATCH \
    virtual void *exec(const type_index &ti, void *in, void *copy, void *buf, \
      bool fwd, size_t nthreads=1) const \
      { \
      static const auto tics = tidx<Tcs *>(); \
      if (ti==tics) \
        { \
        auto in1 = static_cast<Tcs *>(in); \
        auto copy1 = static_cast<Tcs *>(copy); \
        auto buf1 = static_cast<Tcs *>(buf); \
        return fwd ? exec_<true>(in1, copy1, buf1, nthreads) \
                   : exec_<false>(in1, copy1, buf1, nthreads); \
        } \
      if constexpr (fft1d_simdlen<Tfs> > 1) \
        if constexpr (simd_exists<Tfs, fft1d_simdlen<Tfs>>) \
          { \
          using Tfv = typename simd_select<Tfs, fft1d_simdlen<Tfs>>::type; \
          using Tcv = Cmplx<Tfv>; \
          static const auto ticv = tidx<Tcv *>(); \
          if (ti==ticv) \
            {  \
            auto in1 = static_cast<Tcv *>(in); \
            auto copy1 = static_cast<Tcv *>(copy); \
            auto buf1 = static_cast<Tcv *>(buf); \
            return fwd ? exec_<true>(in1, copy1, buf1, nthreads) \
                       : exec_<false>(in1, copy1, buf1, nthreads); \
            } \
          } \
      if constexpr (fft1d_simdlen<Tfs> > 2) \
        if constexpr (simd_exists<Tfs, fft1d_simdlen<Tfs>/2>) \
          { \
          using Tfv = typename simd_select<Tfs, fft1d_simdlen<Tfs>/2>::type; \
          using Tcv = Cmplx<Tfv>; \
          static const auto ticv = tidx<Tcv *>(); \
          if (ti==ticv) \
            {  \
            auto in1 = static_cast<Tcv *>(in); \
            auto copy1 = static_cast<Tcv *>(copy); \
            auto buf1 = static_cast<Tcv *>(buf); \
            return fwd ? exec_<true>(in1, copy1, buf1, nthreads) \
                       : exec_<false>(in1, copy1, buf1, nthreads); \
            } \
          } \
      if constexpr (fft1d_simdlen<Tfs> > 4) \
        if constexpr (simd_exists<Tfs, fft1d_simdlen<Tfs>/4>) \
          { \
          using Tfv = typename simd_select<Tfs, fft1d_simdlen<Tfs>/4>::type; \
          using Tcv = Cmplx<Tfv>; \
          static const auto ticv = tidx<Tcv *>(); \
          if (ti==ticv) \
            {  \
            auto in1 = static_cast<Tcv *>(in); \
            auto copy1 = static_cast<Tcv *>(copy); \
            auto buf1 = static_cast<Tcv *>(buf); \
            return fwd ? exec_<true>(in1, copy1, buf1, nthreads) \
                       : exec_<false>(in1, copy1, buf1, nthreads); \
            } \
          } \
      if constexpr (fft1d_simdlen<Tfs> > 8) \
        if constexpr (simd_exists<Tfs, fft1d_simdlen<Tfs>/8>) \
          { \
          using Tfv = typename simd_select<Tfs, fft1d_simdlen<Tfs>/8>::type; \
          using Tcv = Cmplx<Tfv>; \
          static const auto ticv = tidx<Tcv *>(); \
          if (ti==ticv) \
            {  \
            auto in1 = static_cast<Tcv *>(in); \
            auto copy1 = static_cast<Tcv *>(copy); \
            auto buf1 = static_cast<Tcv *>(buf); \
            return fwd ? exec_<true>(in1, copy1, buf1, nthreads) \
                       : exec_<false>(in1, copy1, buf1, nthreads); \
            } \
          } \
      MR_fail("impossible vector length requested"); \
      }

template <typename Tfs> class cfftp1: public cfftpass<Tfs>
  {
  public:
    cfftp1() {}
    virtual size_t bufsize() const { return 0; }
    virtual bool needs_copy() const { return false; }

    virtual void *exec(const type_index & /*ti*/, void * in, void * /*copy*/,
      void * /*buf*/, bool /*fwd*/, size_t /*nthreads*/) const
      { return in; }
  };

template <typename Tfs> class cfftp2: public cfftpass<Tfs>
  {
  private:
    using typename cfftpass<Tfs>::Tcs;

    size_t l1, ido;
    static constexpr size_t ip=2;
    aligned_array<Tcs> wa;

    auto WA(size_t i) const
      { return wa[i-1]; }

    template<bool fwd, typename Tcd> Tcd *exec_ (const Tcd * DUCC0_RESTRICT cc,
      Tcd * DUCC0_RESTRICT ch, Tcd * /*buf*/, size_t /*nthreads*/) const
      {
      if (ido==1)
        {
        auto CH = [ch,this](size_t b, size_t c) -> Tcd&
          { return ch[b+l1*c]; };
        auto CC = [cc](size_t b, size_t c) -> const Tcd&
          { return cc[b+ip*c]; };
        for (size_t k=0; k<l1; ++k)
          {
          CH(k,0) = CC(0,k)+CC(1,k);
          CH(k,1) = CC(0,k)-CC(1,k);
          }
        return ch;
        }
      else
        {
        auto CH = [ch,this](size_t a, size_t b, size_t c) -> Tcd&
          { return ch[a+ido*(b+l1*c)]; };
        auto CC = [cc,this](size_t a, size_t b, size_t c) -> const Tcd&
          { return cc[a+ido*(b+ip*c)]; };
        for (size_t k=0; k<l1; ++k)
          {
          CH(0,k,0) = CC(0,0,k)+CC(0,1,k);
          CH(0,k,1) = CC(0,0,k)-CC(0,1,k);
          for (size_t i=1; i<ido; ++i)
            {
            CH(i,k,0) = CC(i,0,k)+CC(i,1,k);
            special_mul<fwd>(CC(i,0,k)-CC(i,1,k),WA(i),CH(i,k,1));
            }
          }
        return ch;
        }
      }

  public:
    cfftp2(size_t l1_, size_t ido_, const Troots<Tfs> &roots)
      : l1(l1_), ido(ido_), wa((ip-1)*(ido-1))
      {
      size_t N=ip*l1*ido;
      size_t rfct = roots->size()/N;
      MR_assert(roots->size()==N*rfct, "mismatch");
      for (size_t i=1; i<ido; ++i)
        wa[i-1] = (*roots)[rfct*l1*i];
      }

    virtual size_t bufsize() const { return 0; }
    virtual bool needs_copy() const { return true; }

    POCKETFFT_EXEC_DISPATCH
  };

template <typename Tfs> class cfftp3: public cfftpass<Tfs>
  {
  private:
    using typename cfftpass<Tfs>::Tcs;

    size_t l1, ido;
    static constexpr size_t ip=3;
    aligned_array<Tcs> wa;

    auto WA(size_t x, size_t i) const
      { return wa[x+(i-1)*(ip-1)]; }

    template<bool fwd, typename Tcd> Tcd *exec_
      (const Tcd * DUCC0_RESTRICT cc, Tcd * DUCC0_RESTRICT ch, Tcd * /*buf*/,
      size_t /*nthreads*/) const
      {
      constexpr Tfs tw1r=-0.5,
                    tw1i= (fwd ? -1: 1) * Tfs(0.8660254037844386467637231707529362L);

      auto CH = [ch,this](size_t a, size_t b, size_t c) -> Tcd&
        { return ch[a+ido*(b+l1*c)]; };
      auto CC = [cc,this](size_t a, size_t b, size_t c) -> const Tcd&
        { return cc[a+ido*(b+ip*c)]; };

#define POCKETFFT_PREP3(idx) \
        Tcd t0 = CC(idx,0,k), t1, t2; \
        PM (t1,t2,CC(idx,1,k),CC(idx,2,k)); \
        CH(idx,k,0)=t0+t1;
#define POCKETFFT_PARTSTEP3a(u1,u2,twr,twi) \
        { \
        Tcd ca=t0+t1*twr; \
        Tcd cb{-t2.i*twi, t2.r*twi}; \
        PM(CH(0,k,u1),CH(0,k,u2),ca,cb) ;\
        }
#define POCKETFFT_PARTSTEP3b(u1,u2,twr,twi) \
        { \
        Tcd ca=t0+t1*twr; \
        Tcd cb{-t2.i*twi, t2.r*twi}; \
        special_mul<fwd>(ca+cb,WA(u1-1,i),CH(i,k,u1)); \
        special_mul<fwd>(ca-cb,WA(u2-1,i),CH(i,k,u2)); \
        }

      if (ido==1)
        for (size_t k=0; k<l1; ++k)
          {
          POCKETFFT_PREP3(0)
          POCKETFFT_PARTSTEP3a(1,2,tw1r,tw1i)
          }
      else
        for (size_t k=0; k<l1; ++k)
          {
          {
          POCKETFFT_PREP3(0)
          POCKETFFT_PARTSTEP3a(1,2,tw1r,tw1i)
          }
          for (size_t i=1; i<ido; ++i)
            {
            POCKETFFT_PREP3(i)
            POCKETFFT_PARTSTEP3b(1,2,tw1r,tw1i)
            }
          }

#undef POCKETFFT_PARTSTEP3b
#undef POCKETFFT_PARTSTEP3a
#undef POCKETFFT_PREP3

      return ch;
      }

  public:
    cfftp3(size_t l1_, size_t ido_, const Troots<Tfs> &roots)
      : l1(l1_), ido(ido_), wa((ip-1)*(ido-1))
      {
      size_t N=ip*l1*ido;
      size_t rfct = roots->size()/N;
      MR_assert(roots->size()==N*rfct, "mismatch");
      for (size_t i=1; i<ido; ++i)
        for (size_t j=1; j<ip; ++j)
          wa[(j-1)+(i-1)*(ip-1)] = (*roots)[rfct*j*l1*i];
      }

    virtual size_t bufsize() const { return 0; }
    virtual bool needs_copy() const { return true; }

    POCKETFFT_EXEC_DISPATCH
  };

template <typename Tfs> class cfftp4: public cfftpass<Tfs>
  {
  private:
    using typename cfftpass<Tfs>::Tcs;

    size_t l1, ido;
    static constexpr size_t ip=4;
    aligned_array<Tcs> wa;

    auto WA(size_t x, size_t i) const
      { return wa[x+(i-1)*(ip-1)]; }

    template<bool fwd, typename Tcd> Tcd *exec_
      (const Tcd * DUCC0_RESTRICT cc, Tcd * DUCC0_RESTRICT ch, Tcd * /*buf*/,
      size_t /*nthreads*/) const
      {
      if (ido==1)
        {
        auto CH = [ch,this](size_t b, size_t c) -> Tcd&
          { return ch[b+l1*c]; };
        auto CC = [cc](size_t b, size_t c) -> const Tcd&
          { return cc[b+ip*c]; };
        for (size_t k=0; k<l1; ++k)
          {
          Tcd t1, t2, t3, t4;
          PM(t2,t1,CC(0,k),CC(2,k));
          PM(t3,t4,CC(1,k),CC(3,k));
          ROTX90<fwd>(t4);
          PM(CH(k,0),CH(k,2),t2,t3);
          PM(CH(k,1),CH(k,3),t1,t4);
          }
        }
      else
        {
        auto CH = [ch,this](size_t a, size_t b, size_t c) -> Tcd&
          { return ch[a+ido*(b+l1*c)]; };
        auto CC = [cc,this](size_t a, size_t b, size_t c) -> const Tcd&
          { return cc[a+ido*(b+ip*c)]; };
        for (size_t k=0; k<l1; ++k)
          {
          {
          Tcd t1, t2, t3, t4;
          PM(t2,t1,CC(0,0,k),CC(0,2,k));
          PM(t3,t4,CC(0,1,k),CC(0,3,k));
          ROTX90<fwd>(t4);
          PM(CH(0,k,0),CH(0,k,2),t2,t3);
          PM(CH(0,k,1),CH(0,k,3),t1,t4);
          }
          for (size_t i=1; i<ido; ++i)
            {
            Tcd t1, t2, t3, t4;
            Tcd cc0=CC(i,0,k), cc1=CC(i,1,k),cc2=CC(i,2,k),cc3=CC(i,3,k);
            PM(t2,t1,cc0,cc2);
            PM(t3,t4,cc1,cc3);
            ROTX90<fwd>(t4);
            CH(i,k,0) = t2+t3;
            special_mul<fwd>(t1+t4,WA(0,i),CH(i,k,1));
            special_mul<fwd>(t2-t3,WA(1,i),CH(i,k,2));
            special_mul<fwd>(t1-t4,WA(2,i),CH(i,k,3));
            }
          }
        }
      return ch;
      }

  public:
    cfftp4(size_t l1_, size_t ido_, const Troots<Tfs> &roots)
      : l1(l1_), ido(ido_), wa((ip-1)*(ido-1))
      {
      size_t N=ip*l1*ido;
      size_t rfct = roots->size()/N;
      MR_assert(roots->size()==N*rfct, "mismatch");
      for (size_t i=1; i<ido; ++i)
        for (size_t j=1; j<ip; ++j)
          wa[(j-1)+(i-1)*(ip-1)] = (*roots)[rfct*j*l1*i];
      }

    virtual size_t bufsize() const { return 0; }
    virtual bool needs_copy() const { return true; }

    POCKETFFT_EXEC_DISPATCH
  };

template <typename Tfs> class cfftp5: public cfftpass<Tfs>
  {
  private:
    using typename cfftpass<Tfs>::Tcs;

    size_t l1, ido;
    static constexpr size_t ip=5;
    aligned_array<Tcs> wa;

    auto WA(size_t x, size_t i) const
      { return wa[x+(i-1)*(ip-1)]; }

    template<bool fwd, typename Tcd> Tcd *exec_
      (const Tcd * DUCC0_RESTRICT cc, Tcd * DUCC0_RESTRICT ch, Tcd * /*buf*/,
      size_t /*nthreads*/) const
      {
      constexpr Tfs tw1r= Tfs(0.3090169943749474241022934171828191L),
                    tw1i= (fwd ? -1: 1) * Tfs(0.9510565162951535721164393333793821L),
                    tw2r= Tfs(-0.8090169943749474241022934171828191L),
                    tw2i= (fwd ? -1: 1) * Tfs(0.5877852522924731291687059546390728L);

      auto CH = [ch,this](size_t a, size_t b, size_t c) -> Tcd&
        { return ch[a+ido*(b+l1*c)]; };
      auto CC = [cc,this](size_t a, size_t b, size_t c) -> const Tcd&
        { return cc[a+ido*(b+ip*c)]; };

#define POCKETFFT_PREP5(idx) \
        Tcd t0 = CC(idx,0,k), t1, t2, t3, t4; \
        PM (t1,t4,CC(idx,1,k),CC(idx,4,k)); \
        PM (t2,t3,CC(idx,2,k),CC(idx,3,k)); \
        CH(idx,k,0).r=t0.r+t1.r+t2.r; \
        CH(idx,k,0).i=t0.i+t1.i+t2.i;

#define POCKETFFT_PARTSTEP5a(u1,u2,twar,twbr,twai,twbi) \
        { \
        Tcd ca,cb; \
        ca.r=t0.r+twar*t1.r+twbr*t2.r; \
        ca.i=t0.i+twar*t1.i+twbr*t2.i; \
        cb.i=twai*t4.r twbi*t3.r; \
        cb.r=-(twai*t4.i twbi*t3.i); \
        PM(CH(0,k,u1),CH(0,k,u2),ca,cb); \
        }

#define POCKETFFT_PARTSTEP5b(u1,u2,twar,twbr,twai,twbi) \
        { \
        Tcd ca,cb,da,db; \
        ca.r=t0.r+twar*t1.r+twbr*t2.r; \
        ca.i=t0.i+twar*t1.i+twbr*t2.i; \
        cb.i=twai*t4.r twbi*t3.r; \
        cb.r=-(twai*t4.i twbi*t3.i); \
        special_mul<fwd>(ca+cb,WA(u1-1,i),CH(i,k,u1)); \
        special_mul<fwd>(ca-cb,WA(u2-1,i),CH(i,k,u2)); \
        }

      if (ido==1)
        for (size_t k=0; k<l1; ++k)
          {
          POCKETFFT_PREP5(0)
          POCKETFFT_PARTSTEP5a(1,4,tw1r,tw2r,+tw1i,+tw2i)
          POCKETFFT_PARTSTEP5a(2,3,tw2r,tw1r,+tw2i,-tw1i)
          }
      else
        for (size_t k=0; k<l1; ++k)
          {
          {
          POCKETFFT_PREP5(0)
          POCKETFFT_PARTSTEP5a(1,4,tw1r,tw2r,+tw1i,+tw2i)
          POCKETFFT_PARTSTEP5a(2,3,tw2r,tw1r,+tw2i,-tw1i)
          }
          for (size_t i=1; i<ido; ++i)
            {
            POCKETFFT_PREP5(i)
            POCKETFFT_PARTSTEP5b(1,4,tw1r,tw2r,+tw1i,+tw2i)
            POCKETFFT_PARTSTEP5b(2,3,tw2r,tw1r,+tw2i,-tw1i)
            }
          }

#undef POCKETFFT_PARTSTEP5b
#undef POCKETFFT_PARTSTEP5a
#undef POCKETFFT_PREP5

      return ch;
      }

  public:
    cfftp5(size_t l1_, size_t ido_, const Troots<Tfs> &roots)
      : l1(l1_), ido(ido_), wa((ip-1)*(ido-1))
      {
      size_t N=ip*l1*ido;
      auto rfct = roots->size()/N;
      MR_assert(roots->size()==N*rfct, "mismatch");
      for (size_t i=1; i<ido; ++i)
        for (size_t j=1; j<ip; ++j)
          wa[(j-1)+(i-1)*(ip-1)] = (*roots)[rfct*j*l1*i];
      }

    virtual size_t bufsize() const { return 0; }
    virtual bool needs_copy() const { return true; }

    POCKETFFT_EXEC_DISPATCH
  };

template <typename Tfs> class cfftp7: public cfftpass<Tfs>
  {
  private:
    using typename cfftpass<Tfs>::Tcs;

    size_t l1, ido;
    static constexpr size_t ip=7;
    aligned_array<Tcs> wa;

    auto WA(size_t x, size_t i) const
      { return wa[x+(i-1)*(ip-1)]; }

    template<bool fwd, typename Tcd> Tcd *exec_
      (const Tcd * DUCC0_RESTRICT cc, Tcd * DUCC0_RESTRICT ch, Tcd * /*buf*/,
      size_t /*nthreads*/) const
      {
      constexpr Tfs tw1r= Tfs(0.6234898018587335305250048840042398L),
                    tw1i= (fwd ? -1 : 1) * Tfs(0.7818314824680298087084445266740578L),
                    tw2r= Tfs(-0.2225209339563144042889025644967948L),
                    tw2i= (fwd ? -1 : 1) * Tfs(0.9749279121818236070181316829939312L),
                    tw3r= Tfs(-0.9009688679024191262361023195074451L),
                    tw3i= (fwd ? -1 : 1) * Tfs(0.433883739117558120475768332848359L);

      auto CH = [ch,this](size_t a, size_t b, size_t c) -> Tcd&
        { return ch[a+ido*(b+l1*c)]; };
      auto CC = [cc,this](size_t a, size_t b, size_t c) -> const Tcd&
        { return cc[a+ido*(b+ip*c)]; };

#define POCKETFFT_PREP7(idx) \
        Tcd t1 = CC(idx,0,k), t2, t3, t4, t5, t6, t7; \
        PM (t2,t7,CC(idx,1,k),CC(idx,6,k)); \
        PM (t3,t6,CC(idx,2,k),CC(idx,5,k)); \
        PM (t4,t5,CC(idx,3,k),CC(idx,4,k)); \
        CH(idx,k,0).r=t1.r+t2.r+t3.r+t4.r; \
        CH(idx,k,0).i=t1.i+t2.i+t3.i+t4.i;

#define POCKETFFT_PARTSTEP7a0(u1,u2,x1,x2,x3,y1,y2,y3,out1,out2) \
        { \
        Tcd ca,cb; \
        ca.r=t1.r+x1*t2.r+x2*t3.r+x3*t4.r; \
        ca.i=t1.i+x1*t2.i+x2*t3.i+x3*t4.i; \
        cb.i=y1*t7.r y2*t6.r y3*t5.r; \
        cb.r=-(y1*t7.i y2*t6.i y3*t5.i); \
        PM(out1,out2,ca,cb); \
        }
#define POCKETFFT_PARTSTEP7a(u1,u2,x1,x2,x3,y1,y2,y3) \
        POCKETFFT_PARTSTEP7a0(u1,u2,x1,x2,x3,y1,y2,y3,CH(0,k,u1),CH(0,k,u2))
#define POCKETFFT_PARTSTEP7(u1,u2,x1,x2,x3,y1,y2,y3) \
        { \
        Tcd da,db; \
        POCKETFFT_PARTSTEP7a0(u1,u2,x1,x2,x3,y1,y2,y3,da,db) \
        special_mul<fwd>(da,WA(u1-1,i),CH(i,k,u1)); \
        special_mul<fwd>(db,WA(u2-1,i),CH(i,k,u2)); \
        }

      if (ido==1)
        for (size_t k=0; k<l1; ++k)
          {
          POCKETFFT_PREP7(0)
          POCKETFFT_PARTSTEP7a(1,6,tw1r,tw2r,tw3r,+tw1i,+tw2i,+tw3i)
          POCKETFFT_PARTSTEP7a(2,5,tw2r,tw3r,tw1r,+tw2i,-tw3i,-tw1i)
          POCKETFFT_PARTSTEP7a(3,4,tw3r,tw1r,tw2r,+tw3i,-tw1i,+tw2i)
          }
      else
        for (size_t k=0; k<l1; ++k)
          {
          {
          POCKETFFT_PREP7(0)
          POCKETFFT_PARTSTEP7a(1,6,tw1r,tw2r,tw3r,+tw1i,+tw2i,+tw3i)
          POCKETFFT_PARTSTEP7a(2,5,tw2r,tw3r,tw1r,+tw2i,-tw3i,-tw1i)
          POCKETFFT_PARTSTEP7a(3,4,tw3r,tw1r,tw2r,+tw3i,-tw1i,+tw2i)
          }
          for (size_t i=1; i<ido; ++i)
            {
            POCKETFFT_PREP7(i)
            POCKETFFT_PARTSTEP7(1,6,tw1r,tw2r,tw3r,+tw1i,+tw2i,+tw3i)
            POCKETFFT_PARTSTEP7(2,5,tw2r,tw3r,tw1r,+tw2i,-tw3i,-tw1i)
            POCKETFFT_PARTSTEP7(3,4,tw3r,tw1r,tw2r,+tw3i,-tw1i,+tw2i)
            }
          }

#undef POCKETFFT_PARTSTEP7
#undef POCKETFFT_PARTSTEP7a0
#undef POCKETFFT_PARTSTEP7a
#undef POCKETFFT_PREP7

      return ch;
      }

  public:
    cfftp7(size_t l1_, size_t ido_, const Troots<Tfs> &roots)
      : l1(l1_), ido(ido_), wa((ip-1)*(ido-1))
      {
      size_t N=ip*l1*ido;
      auto rfct = roots->size()/N;
      MR_assert(roots->size()==N*rfct, "mismatch");
      for (size_t i=1; i<ido; ++i)
        for (size_t j=1; j<ip; ++j)
          wa[(j-1)+(i-1)*(ip-1)] = (*roots)[rfct*j*l1*i];
      }

    virtual size_t bufsize() const { return 0; }
    virtual bool needs_copy() const { return true; }

    POCKETFFT_EXEC_DISPATCH
  };

template <typename Tfs> class cfftp8: public cfftpass<Tfs>
  {
  private:
    using typename cfftpass<Tfs>::Tcs;

    size_t l1, ido;
    static constexpr size_t ip=8;
    aligned_array<Tcs> wa;

    auto WA(size_t x, size_t i) const
      { return wa[x+(i-1)*(ip-1)]; }

    template <bool fwd, typename T> void ROTX45(T &a) const
      {
      constexpr Tfs hsqt2=Tfs(0.707106781186547524400844362104849L);
      if constexpr (fwd)
        { auto tmp_=a.r; a.r=hsqt2*(a.r+a.i); a.i=hsqt2*(a.i-tmp_); }
      else
        { auto tmp_=a.r; a.r=hsqt2*(a.r-a.i); a.i=hsqt2*(a.i+tmp_); }
      }
    template <bool fwd, typename T> void ROTX135(T &a) const
      {
      constexpr Tfs hsqt2=Tfs(0.707106781186547524400844362104849L);
      if constexpr (fwd)
        { auto tmp_=a.r; a.r=hsqt2*(a.i-a.r); a.i=hsqt2*(-tmp_-a.i); }
      else
        { auto tmp_=a.r; a.r=hsqt2*(-a.r-a.i); a.i=hsqt2*(tmp_-a.i); }
      }

    template<bool fwd, typename Tcd> Tcd *exec_
      (Tcd * DUCC0_RESTRICT cc, Tcd * DUCC0_RESTRICT ch, Tcd * /*buf*/, size_t /*nthreads*/) const
      {
      if (l1==1)
        {
        auto CC = [cc,this](size_t a, size_t b) -> Tcd&
          { return cc[a+ido*b]; };
        {
        Tcd a0, a1, a2, a3, a4, a5, a6, a7;
        PM(a1,a5,CC(0,1),CC(0,5));
        PM(a3,a7,CC(0,3),CC(0,7));
        PMINPLACE(a1,a3);
        ROTX90<fwd>(a3);

        ROTX90<fwd>(a7);
        PMINPLACE(a5,a7);
        ROTX45<fwd>(a5);
        ROTX135<fwd>(a7);

        PM(a0,a4,CC(0,0),CC(0,4));
        PM(a2,a6,CC(0,2),CC(0,6));
        PM(CC(0,0),CC(0,4),a0+a2,a1);
        PM(CC(0,2),CC(0,6),a0-a2,a3);
        ROTX90<fwd>(a6);
        PM(CC(0,1),CC(0,5),a4+a6,a5);
        PM(CC(0,3),CC(0,7),a4-a6,a7);
        }
        for (size_t i=1; i<ido; ++i)
          {
          Tcd a0, a1, a2, a3, a4, a5, a6, a7;
          PM(a1,a5,CC(i,1),CC(i,5));
          PM(a3,a7,CC(i,3),CC(i,7));
          ROTX90<fwd>(a7);
          PMINPLACE(a1,a3);
          ROTX90<fwd>(a3);
          PMINPLACE(a5,a7);
          ROTX45<fwd>(a5);
          ROTX135<fwd>(a7);
          PM(a0,a4,CC(i,0),CC(i,4));
          PM(a2,a6,CC(i,2),CC(i,6));
          PMINPLACE(a0,a2);
          CC(i,0) = a0+a1;
          special_mul<fwd>(a0-a1,WA(3,i),CC(i,4));
          special_mul<fwd>(a2+a3,WA(1,i),CC(i,2));
          special_mul<fwd>(a2-a3,WA(5,i),CC(i,6));
          ROTX90<fwd>(a6);
          PMINPLACE(a4,a6);
          special_mul<fwd>(a4+a5,WA(0,i),CC(i,1));
          special_mul<fwd>(a4-a5,WA(4,i),CC(i,5));
          special_mul<fwd>(a6+a7,WA(2,i),CC(i,3));
          special_mul<fwd>(a6-a7,WA(6,i),CC(i,7));
          }
        return cc;
        }
      if (ido==1)
        {
        auto CH = [ch,this](size_t b, size_t c) -> Tcd&
          { return ch[b+l1*c]; };
        auto CC = [cc](size_t b, size_t c) -> const Tcd&
          { return cc[b+ip*c]; };
        for (size_t k=0; k<l1; ++k)
          {
          Tcd a0, a1, a2, a3, a4, a5, a6, a7;
          PM(a1,a5,CC(1,k),CC(5,k));
          PM(a3,a7,CC(3,k),CC(7,k));
          PMINPLACE(a1,a3);
          ROTX90<fwd>(a3);

          ROTX90<fwd>(a7);
          PMINPLACE(a5,a7);
          ROTX45<fwd>(a5);
          ROTX135<fwd>(a7);

          PM(a0,a4,CC(0,k),CC(4,k));
          PM(a2,a6,CC(2,k),CC(6,k));
          PM(CH(k,0),CH(k,4),a0+a2,a1);
          PM(CH(k,2),CH(k,6),a0-a2,a3);
          ROTX90<fwd>(a6);
          PM(CH(k,1),CH(k,5),a4+a6,a5);
          PM(CH(k,3),CH(k,7),a4-a6,a7);
          }
        }
      else
        {
        auto CH = [ch,this](size_t a, size_t b, size_t c) -> Tcd&
          { return ch[a+ido*(b+l1*c)]; };
        auto CC = [cc,this](size_t a, size_t b, size_t c) -> const Tcd&
          { return cc[a+ido*(b+ip*c)]; };
        for (size_t k=0; k<l1; ++k)
          {
          {
          Tcd a0, a1, a2, a3, a4, a5, a6, a7;
          PM(a1,a5,CC(0,1,k),CC(0,5,k));
          PM(a3,a7,CC(0,3,k),CC(0,7,k));
          PMINPLACE(a1,a3);
          ROTX90<fwd>(a3);

          ROTX90<fwd>(a7);
          PMINPLACE(a5,a7);
          ROTX45<fwd>(a5);
          ROTX135<fwd>(a7);

          PM(a0,a4,CC(0,0,k),CC(0,4,k));
          PM(a2,a6,CC(0,2,k),CC(0,6,k));
          PM(CH(0,k,0),CH(0,k,4),a0+a2,a1);
          PM(CH(0,k,2),CH(0,k,6),a0-a2,a3);
          ROTX90<fwd>(a6);
          PM(CH(0,k,1),CH(0,k,5),a4+a6,a5);
          PM(CH(0,k,3),CH(0,k,7),a4-a6,a7);
          }
          for (size_t i=1; i<ido; ++i)
            {
            Tcd a0, a1, a2, a3, a4, a5, a6, a7;
            PM(a1,a5,CC(i,1,k),CC(i,5,k));
            PM(a3,a7,CC(i,3,k),CC(i,7,k));
            ROTX90<fwd>(a7);
            PMINPLACE(a1,a3);
            ROTX90<fwd>(a3);
            PMINPLACE(a5,a7);
            ROTX45<fwd>(a5);
            ROTX135<fwd>(a7);
            PM(a0,a4,CC(i,0,k),CC(i,4,k));
            PM(a2,a6,CC(i,2,k),CC(i,6,k));
            PMINPLACE(a0,a2);
            CH(i,k,0) = a0+a1;
            special_mul<fwd>(a0-a1,WA(3,i),CH(i,k,4));
            special_mul<fwd>(a2+a3,WA(1,i),CH(i,k,2));
            special_mul<fwd>(a2-a3,WA(5,i),CH(i,k,6));
            ROTX90<fwd>(a6);
            PMINPLACE(a4,a6);
            special_mul<fwd>(a4+a5,WA(0,i),CH(i,k,1));
            special_mul<fwd>(a4-a5,WA(4,i),CH(i,k,5));
            special_mul<fwd>(a6+a7,WA(2,i),CH(i,k,3));
            special_mul<fwd>(a6-a7,WA(6,i),CH(i,k,7));
            }
          }
        }
      return ch;
      }

  public:
    cfftp8(size_t l1_, size_t ido_, const Troots<Tfs> &roots)
      : l1(l1_), ido(ido_), wa((ip-1)*(ido-1))
      {
      size_t N=ip*l1*ido;
      auto rfct = roots->size()/N;
      MR_assert(roots->size()==N*rfct, "mismatch");
      for (size_t i=1; i<ido; ++i)
        for (size_t j=1; j<ip; ++j)
          wa[(j-1)+(i-1)*(ip-1)] = (*roots)[rfct*j*l1*i];
      }

    virtual size_t bufsize() const { return 0; }
    virtual bool needs_copy() const { return l1>1; }

    POCKETFFT_EXEC_DISPATCH
  };

template <typename Tfs> class cfftp11: public cfftpass<Tfs>
  {
  private:
    using typename cfftpass<Tfs>::Tcs;

    size_t l1, ido;
    static constexpr size_t ip=11;
    aligned_array<Tcs> wa;

    auto WA(size_t x, size_t i) const
      { return wa[x+(i-1)*(ip-1)]; }

    template<bool fwd, typename Tcd> [[gnu::hot]] Tcd *exec_
      (const Tcd * DUCC0_RESTRICT cc, Tcd * DUCC0_RESTRICT ch, Tcd * /*buf*/,
      size_t /*nthreads*/) const
      {
      constexpr Tfs tw1r= Tfs(0.8412535328311811688618116489193677L),
                    tw1i= (fwd ? -1 : 1) * Tfs(0.5406408174555975821076359543186917L),
                    tw2r= Tfs(0.4154150130018864255292741492296232L),
                    tw2i= (fwd ? -1 : 1) * Tfs(0.9096319953545183714117153830790285L),
                    tw3r= Tfs(-0.1423148382732851404437926686163697L),
                    tw3i= (fwd ? -1 : 1) * Tfs(0.9898214418809327323760920377767188L),
                    tw4r= Tfs(-0.6548607339452850640569250724662936L),
                    tw4i= (fwd ? -1 : 1) * Tfs(0.7557495743542582837740358439723444L),
                    tw5r= Tfs(-0.9594929736144973898903680570663277L),
                    tw5i= (fwd ? -1 : 1) * Tfs(0.2817325568414296977114179153466169L);

      auto CH = [ch,this](size_t a, size_t b, size_t c) -> Tcd&
        { return ch[a+ido*(b+l1*c)]; };
      auto CC = [cc,this](size_t a, size_t b, size_t c) -> const Tcd&
        { return cc[a+ido*(b+ip*c)]; };

#define POCKETFFT_PREP11(idx) \
        Tcd t1 = CC(idx,0,k), t2, t3, t4, t5, t6, t7, t8, t9, t10, t11; \
        PM (t2,t11,CC(idx,1,k),CC(idx,10,k)); \
        PM (t3,t10,CC(idx,2,k),CC(idx, 9,k)); \
        PM (t4,t9 ,CC(idx,3,k),CC(idx, 8,k)); \
        PM (t5,t8 ,CC(idx,4,k),CC(idx, 7,k)); \
        PM (t6,t7 ,CC(idx,5,k),CC(idx, 6,k)); \
        CH(idx,k,0).r=t1.r+t2.r+t3.r+t4.r+t5.r+t6.r; \
        CH(idx,k,0).i=t1.i+t2.i+t3.i+t4.i+t5.i+t6.i;

#define POCKETFFT_PARTSTEP11a0(u1,u2,x1,x2,x3,x4,x5,y1,y2,y3,y4,y5,out1,out2) \
        { \
        Tcd ca = t1 + t2*x1 + t3*x2 + t4*x3 + t5*x4 +t6*x5, \
            cb; \
        cb.i=y1*t11.r y2*t10.r y3*t9.r y4*t8.r y5*t7.r; \
        cb.r=-(y1*t11.i y2*t10.i y3*t9.i y4*t8.i y5*t7.i ); \
        PM(out1,out2,ca,cb); \
        }
#define POCKETFFT_PARTSTEP11a(u1,u2,x1,x2,x3,x4,x5,y1,y2,y3,y4,y5) \
        POCKETFFT_PARTSTEP11a0(u1,u2,x1,x2,x3,x4,x5,y1,y2,y3,y4,y5,CH(0,k,u1),CH(0,k,u2))
#define POCKETFFT_PARTSTEP11(u1,u2,x1,x2,x3,x4,x5,y1,y2,y3,y4,y5) \
        { \
        Tcd da,db; \
        POCKETFFT_PARTSTEP11a0(u1,u2,x1,x2,x3,x4,x5,y1,y2,y3,y4,y5,da,db) \
        special_mul<fwd>(da,WA(u1-1,i),CH(i,k,u1)); \
        special_mul<fwd>(db,WA(u2-1,i),CH(i,k,u2)); \
        }

      if (ido==1)
        for (size_t k=0; k<l1; ++k)
          {
          POCKETFFT_PREP11(0)
          POCKETFFT_PARTSTEP11a(1,10,tw1r,tw2r,tw3r,tw4r,tw5r,+tw1i,+tw2i,+tw3i,+tw4i,+tw5i)
          POCKETFFT_PARTSTEP11a(2, 9,tw2r,tw4r,tw5r,tw3r,tw1r,+tw2i,+tw4i,-tw5i,-tw3i,-tw1i)
          POCKETFFT_PARTSTEP11a(3, 8,tw3r,tw5r,tw2r,tw1r,tw4r,+tw3i,-tw5i,-tw2i,+tw1i,+tw4i)
          POCKETFFT_PARTSTEP11a(4, 7,tw4r,tw3r,tw1r,tw5r,tw2r,+tw4i,-tw3i,+tw1i,+tw5i,-tw2i)
          POCKETFFT_PARTSTEP11a(5, 6,tw5r,tw1r,tw4r,tw2r,tw3r,+tw5i,-tw1i,+tw4i,-tw2i,+tw3i)
          }
      else
        for (size_t k=0; k<l1; ++k)
          {
          {
          POCKETFFT_PREP11(0)
          POCKETFFT_PARTSTEP11a(1,10,tw1r,tw2r,tw3r,tw4r,tw5r,+tw1i,+tw2i,+tw3i,+tw4i,+tw5i)
          POCKETFFT_PARTSTEP11a(2, 9,tw2r,tw4r,tw5r,tw3r,tw1r,+tw2i,+tw4i,-tw5i,-tw3i,-tw1i)
          POCKETFFT_PARTSTEP11a(3, 8,tw3r,tw5r,tw2r,tw1r,tw4r,+tw3i,-tw5i,-tw2i,+tw1i,+tw4i)
          POCKETFFT_PARTSTEP11a(4, 7,tw4r,tw3r,tw1r,tw5r,tw2r,+tw4i,-tw3i,+tw1i,+tw5i,-tw2i)
          POCKETFFT_PARTSTEP11a(5, 6,tw5r,tw1r,tw4r,tw2r,tw3r,+tw5i,-tw1i,+tw4i,-tw2i,+tw3i)
          }
          for (size_t i=1; i<ido; ++i)
            {
            POCKETFFT_PREP11(i)
            POCKETFFT_PARTSTEP11(1,10,tw1r,tw2r,tw3r,tw4r,tw5r,+tw1i,+tw2i,+tw3i,+tw4i,+tw5i)
            POCKETFFT_PARTSTEP11(2, 9,tw2r,tw4r,tw5r,tw3r,tw1r,+tw2i,+tw4i,-tw5i,-tw3i,-tw1i)
            POCKETFFT_PARTSTEP11(3, 8,tw3r,tw5r,tw2r,tw1r,tw4r,+tw3i,-tw5i,-tw2i,+tw1i,+tw4i)
            POCKETFFT_PARTSTEP11(4, 7,tw4r,tw3r,tw1r,tw5r,tw2r,+tw4i,-tw3i,+tw1i,+tw5i,-tw2i)
            POCKETFFT_PARTSTEP11(5, 6,tw5r,tw1r,tw4r,tw2r,tw3r,+tw5i,-tw1i,+tw4i,-tw2i,+tw3i)
            }
          }

#undef POCKETFFT_PARTSTEP11
#undef POCKETFFT_PARTSTEP11a0
#undef POCKETFFT_PARTSTEP11a
#undef POCKETFFT_PREP11
      return ch;
      }

  public:
    cfftp11(size_t l1_, size_t ido_, const Troots<Tfs> &roots)
      : l1(l1_), ido(ido_), wa((ip-1)*(ido-1))
      {
      size_t N=ip*l1*ido;
      auto rfct = roots->size()/N;
      MR_assert(roots->size()==N*rfct, "mismatch");
      for (size_t i=1; i<ido; ++i)
        for (size_t j=1; j<ip; ++j)
          wa[(j-1)+(i-1)*(ip-1)] = (*roots)[rfct*j*l1*i];
      }

    virtual size_t bufsize() const { return 0; }
    virtual bool needs_copy() const { return true; }

    POCKETFFT_EXEC_DISPATCH
  };

template <typename Tfs> class cfftpg: public cfftpass<Tfs>
  {
  private:
    using typename cfftpass<Tfs>::Tcs;

    size_t l1, ido;
    size_t ip;
    aligned_array<Tcs> wa;
    aligned_array<Tcs> csarr;

    auto WA(size_t x, size_t i) const
      { return wa[i-1+x*(ido-1)]; }

    template<bool fwd, typename Tcd> Tcd *exec_
      (Tcd * DUCC0_RESTRICT cc, Tcd * DUCC0_RESTRICT ch, Tcd * /*buf*/, size_t /*nthreads*/) const
      {
      size_t ipph = (ip+1)/2;
      size_t idl1 = ido*l1;

      auto CH = [ch,this](size_t a, size_t b, size_t c) -> Tcd&
        { return ch[a+ido*(b+l1*c)]; };
      auto CC = [cc,this](size_t a, size_t b, size_t c) -> const Tcd&
        { return cc[a+ido*(b+ip*c)]; };
      auto CX = [cc,this](size_t a, size_t b, size_t c) -> Tcd&
        { return cc[a+ido*(b+l1*c)]; };
      auto CX2 = [cc, idl1](size_t a, size_t b) -> Tcd&
        { return cc[a+idl1*b]; };
      auto CH2 = [ch, idl1](size_t a, size_t b) -> const Tcd&
        { return ch[a+idl1*b]; };

      for (size_t k=0; k<l1; ++k)
        for (size_t i=0; i<ido; ++i)
          CH(i,k,0) = CC(i,0,k);
      for (size_t j=1, jc=ip-1; j<ipph; ++j, --jc)
        for (size_t k=0; k<l1; ++k)
          for (size_t i=0; i<ido; ++i)
            PM(CH(i,k,j),CH(i,k,jc),CC(i,j,k),CC(i,jc,k));
      for (size_t k=0; k<l1; ++k)
        for (size_t i=0; i<ido; ++i)
          {
          Tcd tmp = CH(i,k,0);
          for (size_t j=1; j<ipph; ++j)
            tmp+=CH(i,k,j);
          CX(i,k,0) = tmp;
          }
      for (size_t l=1, lc=ip-1; l<ipph; ++l, --lc)
        {
        // j=0,1,2
        {
        auto wal  = fwd ? csarr[  l].conj() : csarr[  l];
        auto wal2 = fwd ? csarr[2*l].conj() : csarr[2*l];
        for (size_t ik=0; ik<idl1; ++ik)
          {
          CX2(ik,l ).r = CH2(ik,0).r+wal.r*CH2(ik,1).r+wal2.r*CH2(ik,2).r;
          CX2(ik,l ).i = CH2(ik,0).i+wal.r*CH2(ik,1).i+wal2.r*CH2(ik,2).i;
          CX2(ik,lc).r =-wal.i*CH2(ik,ip-1).i-wal2.i*CH2(ik,ip-2).i;
          CX2(ik,lc).i = wal.i*CH2(ik,ip-1).r+wal2.i*CH2(ik,ip-2).r;
          }
        }

        size_t iwal=2*l;
        size_t j=3, jc=ip-3;
        for (; j<ipph-1; j+=2, jc-=2)
          {
          iwal+=l; if (iwal>ip) iwal-=ip;
          Tcs xwal=fwd ? csarr[iwal].conj() : csarr[iwal];
          iwal+=l; if (iwal>ip) iwal-=ip;
          Tcs xwal2=fwd ? csarr[iwal].conj() : csarr[iwal];
          for (size_t ik=0; ik<idl1; ++ik)
            {
            CX2(ik,l).r += CH2(ik,j).r*xwal.r+CH2(ik,j+1).r*xwal2.r;
            CX2(ik,l).i += CH2(ik,j).i*xwal.r+CH2(ik,j+1).i*xwal2.r;
            CX2(ik,lc).r -= CH2(ik,jc).i*xwal.i+CH2(ik,jc-1).i*xwal2.i;
            CX2(ik,lc).i += CH2(ik,jc).r*xwal.i+CH2(ik,jc-1).r*xwal2.i;
            }
          }
        for (; j<ipph; ++j, --jc)
          {
          iwal+=l; if (iwal>ip) iwal-=ip;
          Tcs xwal=fwd ? csarr[iwal].conj() : csarr[iwal];
          for (size_t ik=0; ik<idl1; ++ik)
            {
            CX2(ik,l).r += CH2(ik,j).r*xwal.r;
            CX2(ik,l).i += CH2(ik,j).i*xwal.r;
            CX2(ik,lc).r -= CH2(ik,jc).i*xwal.i;
            CX2(ik,lc).i += CH2(ik,jc).r*xwal.i;
            }
          }
        }

      // shuffling and twiddling
      if (ido==1)
        for (size_t j=1, jc=ip-1; j<ipph; ++j, --jc)
          for (size_t ik=0; ik<idl1; ++ik)
            {
            Tcd t1=CX2(ik,j), t2=CX2(ik,jc);
            PM(CX2(ik,j),CX2(ik,jc),t1,t2);
            }
      else
        {
        for (size_t j=1, jc=ip-1; j<ipph; ++j,--jc)
          for (size_t k=0; k<l1; ++k)
            {
            Tcd t1=CX(0,k,j), t2=CX(0,k,jc);
            PM(CX(0,k,j),CX(0,k,jc),t1,t2);
            for (size_t i=1; i<ido; ++i)
              {
              Tcd x1, x2;
              PM(x1,x2,CX(i,k,j),CX(i,k,jc));
              size_t idij=(j-1)*(ido-1)+i-1;
              special_mul<fwd>(x1,wa[idij],CX(i,k,j));
              idij=(jc-1)*(ido-1)+i-1;
              special_mul<fwd>(x2,wa[idij],CX(i,k,jc));
              }
            }
        }
      return cc;
      }

  public:
    cfftpg(size_t l1_, size_t ido_, size_t ip_, const Troots<Tfs> &roots)
      : l1(l1_), ido(ido_), ip(ip_), wa((ip-1)*(ido-1)), csarr(ip)
      {
      MR_assert((ip&1)&&(ip>=5), "need an odd number >=5");
      size_t N=ip*l1*ido;
      auto rfct = roots->size()/N;
      MR_assert(roots->size()==N*rfct, "mismatch");
      for (size_t j=1; j<ip; ++j)
        for (size_t i=1; i<ido; ++i)
          wa[(j-1)*(ido-1)+i-1] = (*roots)[rfct*j*l1*i];
      for (size_t i=0; i<ip; ++i)
        csarr[i] = (*roots)[rfct*ido*l1*i];
      }

    virtual size_t bufsize() const { return 0; }
    virtual bool needs_copy() const { return true; }

    POCKETFFT_EXEC_DISPATCH
  };

template <typename Tfs> class cfftpblue: public cfftpass<Tfs>
  {
  private:
    using typename cfftpass<Tfs>::Tcs;

    const size_t l1, ido, ip;
    const size_t ip2;
    const Tcpass<Tfs> subplan;
    aligned_array<Tcs> wa, bk, bkf;
    size_t bufsz;
    bool need_cpy;

    auto WA(size_t x, size_t i) const
      { return wa[i-1+x*(ido-1)]; }

    template<bool fwd, typename Tcd> Tcd *exec_
      (Tcd * DUCC0_RESTRICT cc, Tcd * DUCC0_RESTRICT ch,
       Tcd * DUCC0_RESTRICT buf, size_t nthreads) const
      {
      static const auto ti=tidx<Tcd *>();
      Tcd *akf = &buf[0];
      Tcd *akf2 = subplan->needs_copy() ? (&buf[ip2]) : akf;
      Tcd *subbuf = akf2+ip2;

      auto CH = [ch,this](size_t a, size_t b, size_t c) -> Tcd&
        { return ch[a+ido*(b+l1*c)]; };
      auto CC = [cc,this](size_t a, size_t b, size_t c) -> Tcd&
        { return cc[a+ido*(b+ip*c)]; };

//FIXME: parallelize here?
      for (size_t k=0; k<l1; ++k)
        for (size_t i=0; i<ido; ++i)
          {
          /* initialize a_k and FFT it */
          for (size_t m=0; m<ip; ++m)
            special_mul<fwd>(CC(i,m,k),bk[m],akf[m]);
          auto zero = akf[0]*Tfs(0);
          for (size_t m=ip; m<ip2; ++m)
            akf[m]=zero;

          auto res = static_cast<Tcd *>(subplan->exec(ti,akf,akf2,
            subbuf, true, nthreads));

          /* do the convolution */
          res[0] = res[0].template special_mul<!fwd>(bkf[0]);
          for (size_t m=1; m<(ip2+1)/2; ++m)
            {
            res[m] = res[m].template special_mul<!fwd>(bkf[m]);
            res[ip2-m] = res[ip2-m].template special_mul<!fwd>(bkf[m]);
            }
          if ((ip2&1)==0)
            res[ip2/2] = res[ip2/2].template special_mul<!fwd>(bkf[ip2/2]);

          /* inverse FFT */
          res = static_cast<Tcd *>(subplan->exec(ti, res,
            (res==akf) ? akf2 : akf, subbuf, false, nthreads));

          /* multiply by b_k and write to output buffer */
          if (l1>1)
            {
            if (i==0)
              for (size_t m=0; m<ip; ++m)
                CH(0,k,m) = res[m].template special_mul<fwd>(bk[m]);
            else
              {
              CH(i,k,0) = res[0].template special_mul<fwd>(bk[0]);
              for (size_t m=1; m<ip; ++m)
                CH(i,k,m) = res[m].template special_mul<fwd>(bk[m]*WA(m-1,i));
              }
            }
          else
            {
            if (i==0)
              for (size_t m=0; m<ip; ++m)
                CC(0,m,0) = res[m].template special_mul<fwd>(bk[m]);
            else
              {
              CC(i,0,0) = res[0].template special_mul<fwd>(bk[0]);
              for (size_t m=1; m<ip; ++m)
                CC(i,m,0) = res[m].template special_mul<fwd>(bk[m]*WA(m-1,i));
              }
            }
          }

      return (l1>1) ? ch : cc;
      }

  public:
    cfftpblue(size_t l1_, size_t ido_, size_t ip_, const Troots<Tfs> &roots,
      bool vectorize=false)
      : l1(l1_), ido(ido_), ip(ip_), ip2(util1d::good_size_cmplx(ip*2-1)),
        subplan(cfftpass<Tfs>::make_pass(ip2, vectorize)), wa((ip-1)*(ido-1)),
        bk(ip), bkf(ip2/2+1)
      {
      size_t N=ip*l1*ido;
      auto rfct = roots->size()/N;
      MR_assert(roots->size()==N*rfct, "mismatch");
      for (size_t j=1; j<ip; ++j)
        for (size_t i=1; i<ido; ++i)
          wa[(j-1)*(ido-1)+i-1] = (*roots)[rfct*j*l1*i];

      /* initialize b_k */
      bk[0].Set(1, 0);
      size_t coeff=0;
      auto roots2 = ((roots->size()/(2*ip))*2*ip==roots->size()) ?
                    roots : make_shared<const UnityRoots<Tfs,Tcs>>(2*ip);
      size_t rfct2 = roots2->size()/(2*ip);
      for (size_t m=1; m<ip; ++m)
        {
        coeff+=2*m-1;
        if (coeff>=2*ip) coeff-=2*ip;
        bk[m] = (*roots2)[coeff*rfct2];
        }

      /* initialize the zero-padded, Fourier transformed b_k. Add normalisation. */
      aligned_array<Tcs> tbkf(ip2), tbkf2(ip2);
      Tfs xn2 = Tfs(1)/Tfs(ip2);
      tbkf[0] = bk[0]*xn2;
      for (size_t m=1; m<ip; ++m)
        tbkf[m] = tbkf[ip2-m] = bk[m]*xn2;
      for (size_t m=ip;m<=(ip2-ip);++m)
        tbkf[m].Set(0.,0.);
      aligned_array<Tcs> buf(subplan->bufsize());
      static const auto tics=tidx<Tcs *>();
      auto res = static_cast<Tcs *>(subplan->exec(tics, tbkf.data(),
        tbkf2.data(), buf.data(), true));
      for (size_t i=0; i<ip2/2+1; ++i)
        bkf[i] = res[i];

      need_cpy = l1>1;
      bufsz = ip2*(1+subplan->needs_copy()) + subplan->bufsize();
      }

    virtual size_t bufsize() const { return bufsz; }
    virtual bool needs_copy() const { return need_cpy; }

    POCKETFFT_EXEC_DISPATCH
  };

template <typename Tfs> class cfft_multipass: public cfftpass<Tfs>
  {
  private:
    using typename cfftpass<Tfs>::Tcs;
    static constexpr size_t bunchsize=8;

    const size_t l1, ido;
    size_t ip;
    vector<Tcpass<Tfs>> passes;
    size_t bufsz;
    bool need_cpy;
    size_t rfct;
    Troots<Tfs> myroots;

// FIXME split into sub-functions. This is too long!
    template<bool fwd, typename T> Cmplx<T> *exec_(Cmplx<T> *cc, Cmplx<T> *ch,
      Cmplx<T> *buf, size_t nthreads) const
      {
      using Tc = Cmplx<T>;
      if ((l1==1) && (ido==1)) // no chance at vectorizing
        {
        static const auto tic=tidx<Tc *>();
        Tc *p1=cc, *p2=ch;
        for(const auto &pass: passes)
          {
          auto res = static_cast<Tc *>(pass->exec(tic, p1, p2, buf,
            fwd, nthreads));
          if (res==p2) swap (p1,p2);
          }
        return p1;
        }
      else
        {
        if constexpr(is_same<T,Tfs>::value && fft1d_simd_exists<Tfs>) // we can vectorize!
          {
          using Tfv = fft1d_simd<Tfs>;
          using Tcv = Cmplx<Tfv>;
          constexpr size_t vlen = Tfv::size();
          size_t nvtrans = (l1*ido + vlen-1)/vlen;
          // NOTE: removed "static" here, because it leads to trouble with gcc 7
          // static const type_index ticv = tidx<Tcv *>();
          const type_index ticv = tidx<Tcv *>();

          if (ido==1)
            {
            auto CH = [ch,this](size_t b, size_t c) -> Tc&
              { return ch[b+l1*c]; };
            auto CC = [cc,this](size_t b, size_t c) -> Tc&
              { return cc[b+ip*c]; };

            execStatic(nvtrans, nthreads, 0, [&](auto &sched)
              {
              aligned_array<Tcv> tbuf(2*ip+32+bufsize());
              auto cc2 = &tbuf[0];
              auto ch2 = &tbuf[ip+16];
              auto buf2 = &tbuf[2*ip+32];

              while (auto rng=sched.getNext())
                for(auto itrans=rng.lo; itrans<rng.hi; ++itrans)
                  {
                  for (size_t n=0; n<vlen; ++n)
                    for (size_t m=0; m<ip; ++m)
                      {
                      size_t k = min(l1-1, itrans*vlen+n);
                      cc2[m].r[n] = CC(m,k).r;
                      cc2[m].i[n] = CC(m,k).i;
                      }

                  Tcv *p1=cc2, *p2=ch2;
                  for(const auto &pass: passes)
                    {
                    auto res = static_cast<Tcv *>(pass->exec(ticv,
                      p1, p2, buf2, fwd));
                    if (res==p2) swap (p1,p2);
                    }

                  for (size_t m=0; m<ip; ++m)
                    for (size_t n=0; n<vlen; ++n)
                      {
                      auto k = min(l1-1, itrans*vlen+n);
                      CH(k,m) = { p1[m].r[n], p1[m].i[n] };
                      }
                  }
              });
            return ch;
            }

          if (l1==1)
            {
            auto CC = [cc,this](size_t a, size_t b) -> Tc&
              { return cc[a+ido*b]; };

            execStatic(nvtrans, nthreads, 0, [&](auto &sched)
              {
              aligned_array<Tcv> tbuf(2*ip+32+bufsize());
              auto cc2 = &tbuf[0];
              auto ch2 = &tbuf[ip+16];
              auto buf2 = &tbuf[2*ip+32];

              while (auto rng=sched.getNext())
                for(auto itrans=rng.lo; itrans<rng.hi; ++itrans)
                  {
                  for (size_t m=0; m<ip; ++m)
                    for (size_t n=0; n<vlen; ++n)
                      {
                      size_t i = min(ido-1, itrans*vlen+n);
                      cc2[m].r[n] = CC(i,m).r;
                      cc2[m].i[n] = CC(i,m).i;
                      }

                  Tcv *p1=cc2, *p2=ch2;
                  for(const auto &pass: passes)
                    {
                    auto res = static_cast<Tcv *>(pass->exec(ticv,
                      p1, p2, buf2, fwd));
                    if (res==p2) swap (p1,p2);
                    }

                  for (size_t m=0; m<ip; ++m)
                    for (size_t n=0; n<vlen; ++n)
                      {
                      auto i = itrans*vlen+n;
                      if (i >= ido) break;
                      if (i==0)
                        CC(0,m) = { p1[m].r[n], p1[m].i[n] };
                      else
                        {
                        if (m==0)
                          CC(i,0) = { p1[0].r[n], p1[0].i[n] } ;
                        else
                          CC(i,m) = Tcs(p1[m].r[n],p1[m].i[n]).template special_mul<fwd>((*myroots)[rfct*m*i]);
                        }
                      }
                  }
              });
            return cc;
            }

MR_fail("must not get here");
#if 0
//FIXME this code path is currently unused
          aligned_array<Tcv> tbuf(2*ip+bufsize());
          auto cc2 = &tbuf[0];
          auto ch2 = &tbuf[ip];
          auto buf2 = &tbuf[2*ip];

          auto CH = [ch,this](size_t a, size_t b, size_t c) -> Tc&
            { return ch[a+ido*(b+l1*c)]; };
          auto CC = [cc,this](size_t a, size_t b, size_t c) -> Tc&
            { return cc[a+ido*(b+ip*c)]; };

//FIXME parallelize?
          for (size_t itrans=0; itrans<nvtrans; ++itrans)
            {
            array<size_t, vlen> ix, kx;
            size_t ixcur = (itrans*vlen)%ido;
            size_t kxcur = (itrans*vlen)/ido;
            for (size_t n=0; n<vlen; ++n)
              {
              ix[n] = ixcur;
              kx[n] = min(l1-1,kxcur);
              if (++ixcur==ido)
                {
                ixcur=0;
                ++kxcur;
                }
              }

            for (size_t m=0; m<ip; ++m)
              for (size_t n=0; n<vlen; ++n)
                {
                cc2[m].r[n] = CC(ix[n],m,kx[n]).r;
                cc2[m].i[n] = CC(ix[n],m,kx[n]).i;
                }

            Tcv *p1=cc2, *p2=ch2;
            for(const auto &pass: passes)
              {
              auto res = static_cast<Tcv *>(pass->exec(ticv,
                p1, p2, buf2, fwd));
              if (res==p2) swap (p1,p2);
              }

            for (size_t m=0; m<ip; ++m)
              for (size_t n=0; n<vlen; ++n)
                {
                auto i = ix[n];
                auto k = kx[n];
                if (itrans*vlen+n >= l1*ido) break;
                if (i==0)
                  CH(0,k,m) = { p1[m].r[n], p1[m].i[n] };
                else
                  {
                  if (m==0)
                    CH(i,k,0) = { p1[0].r[n], p1[0].i[n] } ;
                  else
                    CH(i,k,m) = Tcs(p1[m].r[n],p1[m].i[n]).template special_mul<fwd>((*myroots)[rfct*l1*m*i]);
                  }
                }
            }
          return ch;
#endif
          }
        else
          {
          static const auto tic = tidx<Cmplx<T> *>();
          if (ido==1)
            {
// parallelize here!
            for (size_t n=0; n<l1; ++n)
              {
              Cmplx<T> *p1=&cc[n*ip], *p2=ch;
              Cmplx<T> *res = nullptr;
              for(const auto &pass: passes)
                {
                res = static_cast<Cmplx<T> *>(pass->exec(tic,
                  p1, p2, buf, fwd));
                if (res==p2) swap (p1,p2);
                }
              if (res != &cc[n*ip])
                copy(res, res+ip, cc+n*ip);
              }
            // transpose
            size_t nbunch = (l1*ido + bunchsize-1)/bunchsize;
// parallelize here!
            for (size_t ibunch=0; ibunch<nbunch; ++ibunch)
              {
              size_t ntrans = min(bunchsize, l1-ibunch*bunchsize);
              for (size_t m=0; m<ip; ++m)
                for (size_t n=0; n<ntrans; ++n)
                  {
                  size_t itrans = ibunch*bunchsize + n;
                  ch[itrans+m*l1] = cc[m+itrans*ip];
                  }
              }
            return ch;
            }
          if (l1==1)
            {
            auto cc2 = &buf[0];
            auto ch2 = &buf[bunchsize*ip];
            auto buf2 = &buf[(bunchsize+1)*ip];
            size_t nbunch = (ido + bunchsize-1)/bunchsize;

            auto CC = [cc,this](size_t a, size_t b) -> Tc&
              { return cc[a+ido*b]; };

// parallelize here!
             for (size_t ibunch=0; ibunch<nbunch; ++ibunch)
              {
              size_t ntrans = min(bunchsize, ido-ibunch*bunchsize);

              for (size_t m=0; m<ip; ++m)
                for (size_t n=0; n<ntrans; ++n)
                  cc2[m+n*ip] = CC(n+ibunch*bunchsize,m);

              for (size_t n=0; n<ntrans; ++n)
                {
                auto i = n+ibunch*bunchsize;
                Cmplx<T> *p1=&cc2[n*ip], *p2=ch2;
                Cmplx<T> *res = nullptr;
                for(const auto &pass: passes)
                  {
                  res = static_cast<Cmplx<T> *>(pass->exec(tic,
                    p1, p2, buf2, fwd));
                  if (res==p2) swap (p1,p2);
                  }
                if (res==&cc2[n*ip]) // no copying necessary
                  {
                  if (i!=0)
                    {
                    for (size_t m=1; m<ip; ++m)
                      cc2[n*ip+m] = cc2[n*ip+m].template special_mul<fwd>((*myroots)[rfct*m*i]);
                    }
                  }
                else
                  {
                  if (i==0)
                    for (size_t m=0; m<ip; ++m)
                      cc2[n*ip+m] = res[m];
                  else
                    {
                    cc2[n*ip] = res[0];
                    for (size_t m=1; m<ip; ++m)
                      cc2[n*ip+m] = res[m].template special_mul<fwd>((*myroots)[rfct*m*i]);
                    }
                  }
                }
              for (size_t m=0; m<ip; ++m)
                for (size_t n=0; n<ntrans; ++n)
                  CC(n+ibunch*bunchsize, m) = cc2[m+n*ip];
              }
            return cc;
            }

MR_fail("must not get here");
#if 0
//FIXME this code path is currently unused
          auto cc2 = &buf[0];
          auto ch2 = &buf[bunchsize*ip];
          auto buf2 = &buf[(bunchsize+1)*ip];
          size_t nbunch = (l1*ido + bunchsize-1)/bunchsize;

          auto CH = [ch,this](size_t a, size_t b, size_t c) -> Tc&
            { return ch[a+ido*(b+l1*c)]; };
          auto CC = [cc,this](size_t a, size_t b, size_t c) -> Tc&
            { return cc[a+ido*(b+ip*c)]; };

// parallelize here!
          for (size_t ibunch=0; ibunch<nbunch; ++ibunch)
            {
            size_t ntrans = min(bunchsize, l1*ido-ibunch*bunchsize);
            array<size_t, bunchsize> ix, kx;
            size_t ixcur = (ibunch*bunchsize)%ido;
            size_t kxcur = (ibunch*bunchsize)/ido;
            for (size_t n=0; n<bunchsize; ++n)
              {
              ix[n] = ixcur;
              kx[n] = min(l1-1,kxcur);
              if (++ixcur==ido)
                {
                ixcur=0;
                ++kxcur;
                }
              }
            for (size_t m=0; m<ip; ++m)
              for (size_t n=0; n<ntrans; ++n)
                cc2[m+n*ip] = CC(ix[n],m,kx[n]);

            for (size_t n=0; n<ntrans; ++n)
              {
              auto i = ix[n];
              Cmplx<T> *p1=&cc2[n*ip], *p2=ch2;
              Cmplx<T> *res = nullptr;
              for(const auto &pass: passes)
                {
                res = static_cast<Cmplx<T> *>(pass->exec(tic,
                  p1, p2, buf2, fwd));
                if (res==p2) swap (p1,p2);
                }
              if (res==&cc2[n*ip]) // no copying necessary
                {
                if (i!=0)
                  {
                  for (size_t m=1; m<ip; ++m)
                    cc2[n*ip+m] = cc2[n*ip+m].template special_mul<fwd>((*myroots)[rfct*l1*m*i]);
                  }
                }
              else
                {
                if (i==0)
                  for (size_t m=0; m<ip; ++m)
                    cc2[n*ip+m] = res[m];
                else
                  {
                  cc2[n*ip] = res[0];
                  for (size_t m=1; m<ip; ++m)
                    cc2[n*ip+m] = res[m].template special_mul<fwd>((*myroots)[rfct*l1*m*i]);
                  }
                }
              }
            for (size_t m=0; m<ip; ++m)
              for (size_t n=0; n<ntrans; ++n)
                CH(ix[n], kx[n], m) = cc2[m+n*ip];
            }
          return ch;
#endif
          }
        }
      }

  public:
    cfft_multipass(size_t l1_, size_t ido_, size_t ip_,
      const Troots<Tfs> &roots, bool /*vectorize*/=false)
      : l1(l1_), ido(ido_), ip(ip_), bufsz(0), need_cpy(false),
        myroots(roots)
      {
      size_t N=ip*l1*ido;
      rfct = roots->size()/N;
      MR_assert(roots->size()==N*rfct, "mismatch");

      // FIXME TBD
// do we need the vectorize flag at all?
      size_t lim = 10000; //vectorize ? 10000 : 10000;
      if (ip<=lim)
        {
        auto factors = cfftpass<Tfs>::factorize(ip);
        size_t l1l=1;
        for (auto fct: factors)
          {
          passes.push_back(cfftpass<Tfs>::make_pass(l1l, ip/(fct*l1l), fct, roots, false));
          l1l*=fct;
          }
        }
      else
        {
        vector<size_t> packets(2,1);
        auto factors = util1d::prime_factors(ip);
        sort(factors.begin(), factors.end(), std::greater<size_t>());
        for (auto fct: factors)
          (packets[0]>packets[1]) ? packets[1]*=fct : packets[0]*=fct;
        size_t l1l=1;
        for (auto pkt: packets)
          {
          passes.push_back(cfftpass<Tfs>::make_pass(l1l, ip/(pkt*l1l), pkt, roots, false));
          l1l*=pkt;
          }
        }
      for (const auto &pass: passes)
        {
        bufsz = max(bufsz, pass->bufsize());
        need_cpy |= pass->needs_copy();
        }
      if ((l1!=1)||(ido!=1))
        {
        need_cpy=true;
        bufsz += (bunchsize+1)*ip;
        }
      }

    virtual size_t bufsize() const { return bufsz; }
    virtual bool needs_copy() const { return need_cpy; }

    POCKETFFT_EXEC_DISPATCH
  };

#undef POCKETFFT_EXEC_DISPATCH

template <size_t vlen, typename Tfs> class cfftp_vecpass: public cfftpass<Tfs>
  {
  private:
    static_assert(simd_exists<Tfs, vlen>, "bad vlen");
    using typename cfftpass<Tfs>::Tcs;
    using Tfv=typename simd_select<Tfs, vlen>::type;
    using Tcv=Cmplx<Tfv>;

    size_t ip;
    Tcpass<Tfs> spass;
    Tcpass<Tfs> vpass;
    size_t bufsz;

    template<bool fwd> Tcs *exec_ (Tcs *cc,
      Tcs * /*ch*/, Tcs *sbuf, size_t nthreads) const
      {
      char *xbuf = reinterpret_cast<char *>(sbuf);
      size_t misalign = reinterpret_cast<size_t>(xbuf)&(sizeof(Tfv)-1);
      if (misalign != 0)
        xbuf += sizeof(Tfv)-misalign;
      Tcv *buf = reinterpret_cast<Tcv *>(xbuf);
      auto * cc2 = buf;
      auto * ch2 = buf+ip/vlen+7;
      auto * buf2 = buf+2*ip/vlen+7+7;
      static const auto tics = tidx<Tcs *>();
// run scalar pass
      auto res = static_cast<Tcs *>(spass->exec(tics, cc,
        reinterpret_cast<Tcs *>(ch2), reinterpret_cast<Tcs *>(buf2),
        fwd, nthreads));
// arrange input in SIMD-friendly way, must be done out-of-place
      for (size_t i=0; i<ip/vlen; ++i)
        {
        Tcv tmp;
        for (size_t j=0; j<vlen; ++j)
          {
          size_t idx = j*(ip/vlen) + i;
          tmp.r[j] = res[idx].r;
          tmp.i[j] = res[idx].i;
          }
        cc2[i] = tmp;
        }
// run vector pass
      static const auto ticv = tidx<Tcv *>();
      auto res2 = static_cast<Tcv *>(vpass->exec(ticv,
        cc2, ch2, buf2, fwd, nthreads));
// de-SIMDify, can be done pseudo-inplace
      for (size_t i=0; i<ip/vlen; ++i)
        {
        Tcv tmp = res2[i];
        for (size_t j=0; j<vlen; ++j)
          cc[i*vlen+j] = Tcs(tmp.r[j], tmp.i[j]);
        }
      return cc;
      }

  public:
    cfftp_vecpass(size_t ip_, const Troots<Tfs> &roots)
      : ip(ip_), spass(cfftpass<Tfs>::make_pass(1, ip/vlen, vlen, roots)),
        vpass(cfftpass<Tfs>::make_pass(1, 1, ip/vlen, roots)), bufsz(0)
      {
      MR_assert((ip/vlen)*vlen==ip, "cannot vectorize this size");
      bufsz = 2*(ip/vlen)+7+7;
      bufsz += max(vpass->bufsize(),(spass->bufsize()+vlen-1)/vlen); // buffers for subpasses
      bufsz *= vlen; // since we specify in terms of Tcs
      bufsz += vlen; // wiggle room for alignment shifts
      }
    virtual size_t bufsize() const { return bufsz; }
    virtual bool needs_copy() const { return false; }
    virtual void *exec(const type_index &ti, void *in, void *copy, void *buf,
      bool fwd, size_t nthreads=1) const
      {
      static const auto tics = tidx<Tcs *>();
      MR_assert(ti==tics, "bad input type");
      auto in1 = static_cast<Tcs *>(in);
      auto copy1 = static_cast<Tcs *>(copy);
      auto buf1 = static_cast<Tcs *>(buf);
      return fwd ? exec_<true>(in1, copy1, buf1, nthreads)
                 : exec_<false>(in1, copy1, buf1, nthreads);
      }
  };

template<typename Tfs> Tcpass<Tfs> cfftpass<Tfs>::make_pass(size_t l1,
  size_t ido, size_t ip, const Troots<Tfs> &roots, bool vectorize)
  {
  MR_assert(ip>=1, "no zero-sized FFTs");
  // do we have an 1D vectorizable FFT?
  if (vectorize && (ip>300)&& (ip<=100000) && (l1==1) && (ido==1))
    {
//    constexpr auto vlen = native_simd<Tfs>::size();
//    if constexpr(vlen>=4)
    constexpr auto vlen = 4;
    if constexpr(simd_exists<Tfs,vlen>)
      if ((ip&(vlen-1))==0)
        return make_shared<cfftp_vecpass<vlen,Tfs>>(ip, roots);
    }

  if (ip==1) return make_shared<cfftp1<Tfs>>();
  auto factors=cfftpass<Tfs>::factorize(ip);
  if (factors.size()==1)
    {
    switch(ip)
      {
      case 2:
        return make_shared<cfftp2<Tfs>>(l1, ido, roots);
      case 3:
        return make_shared<cfftp3<Tfs>>(l1, ido, roots);
      case 4:
        return make_shared<cfftp4<Tfs>>(l1, ido, roots);
      case 5:
        return make_shared<cfftp5<Tfs>>(l1, ido, roots);
      case 7:
        return make_shared<cfftp7<Tfs>>(l1, ido, roots);
      case 8:
        return make_shared<cfftp8<Tfs>>(l1, ido, roots);
      case 11:
        return make_shared<cfftp11<Tfs>>(l1, ido, roots);
      default:
        if (ip<110)
          return make_shared<cfftpg<Tfs>>(l1, ido, ip, roots);
        else
          return make_shared<cfftpblue<Tfs>>(l1, ido, ip, roots, vectorize);
      }
    }
  else // more than one factor, need a multipass
    return make_shared<cfft_multipass<Tfs>>(l1, ido, ip, roots, vectorize);
  }

#define POCKETFFT_EXEC_DISPATCH \
    virtual void *exec(const type_index &ti, void *in, void *copy, void *buf, \
      bool fwd, size_t nthreads) const \
      { \
      static const auto tifs=tidx<Tfs *>(); \
      if (ti==tifs) \
        { \
        auto in1 = static_cast<Tfs *>(in); \
        auto copy1 = static_cast<Tfs *>(copy); \
        auto buf1 = static_cast<Tfs *>(buf); \
        return fwd ? exec_<true>(in1, copy1, buf1, nthreads) \
                   : exec_<false>(in1, copy1, buf1, nthreads); \
        } \
      if constexpr (fft1d_simdlen<Tfs> > 1) \
        if constexpr (simd_exists<Tfs, fft1d_simdlen<Tfs>>) \
          { \
          using Tfv = typename simd_select<Tfs, fft1d_simdlen<Tfs>>::type; \
          static const auto tifv=tidx<Tfv *>(); \
          if (ti==tifv) \
            {  \
            auto in1 = static_cast<Tfv *>(in); \
            auto copy1 = static_cast<Tfv *>(copy); \
            auto buf1 = static_cast<Tfv *>(buf); \
            return fwd ? exec_<true>(in1, copy1, buf1, nthreads) \
                       : exec_<false>(in1, copy1, buf1, nthreads); \
            } \
          } \
      if constexpr (fft1d_simdlen<Tfs> > 2) \
        if constexpr (simd_exists<Tfs, fft1d_simdlen<Tfs>/2>) \
          { \
          using Tfv = typename simd_select<Tfs, fft1d_simdlen<Tfs>/2>::type; \
          static const auto tifv=tidx<Tfv *>(); \
          if (ti==tifv) \
            {  \
            auto in1 = static_cast<Tfv *>(in); \
            auto copy1 = static_cast<Tfv *>(copy); \
            auto buf1 = static_cast<Tfv *>(buf); \
            return fwd ? exec_<true>(in1, copy1, buf1, nthreads) \
                       : exec_<false>(in1, copy1, buf1, nthreads); \
            } \
          } \
      if constexpr (fft1d_simdlen<Tfs> > 4) \
        if constexpr (simd_exists<Tfs, fft1d_simdlen<Tfs>/4>) \
          { \
          using Tfv = typename simd_select<Tfs, fft1d_simdlen<Tfs>/4>::type; \
          static const auto tifv=tidx<Tfv *>(); \
          if (ti==tifv) \
            {  \
            auto in1 = static_cast<Tfv *>(in); \
            auto copy1 = static_cast<Tfv *>(copy); \
            auto buf1 = static_cast<Tfv *>(buf); \
            return fwd ? exec_<true>(in1, copy1, buf1, nthreads) \
                       : exec_<false>(in1, copy1, buf1, nthreads); \
            } \
          } \
      if constexpr (fft1d_simdlen<Tfs> > 8) \
        if constexpr (simd_exists<Tfs, fft1d_simdlen<Tfs>/8>) \
          { \
          using Tfv = typename simd_select<Tfs, fft1d_simdlen<Tfs>/8>::type; \
          static const auto tifv=tidx<Tfv *>(); \
          if (ti==tifv) \
            {  \
            auto in1 = static_cast<Tfv *>(in); \
            auto copy1 = static_cast<Tfv *>(copy); \
            auto buf1 = static_cast<Tfv *>(buf); \
            return fwd ? exec_<true>(in1, copy1, buf1, nthreads) \
                       : exec_<false>(in1, copy1, buf1, nthreads); \
            } \
          } \
      MR_fail("impossible vector length requested"); \
      }

/* (a+ib) = conj(c+id) * (e+if) */
template<typename T1, typename T2, typename T3> inline void MULPM
  (T1 &a, T1 &b, T2 c, T2 d, T3 e, T3 f)
  {  a=c*e+d*f; b=c*f-d*e; }

template <typename Tfs> class rfftp1: public rfftpass<Tfs>
  {
  public:
    rfftp1() {}
    virtual size_t bufsize() const { return 0; }
    virtual bool needs_copy() const { return false; }

    virtual void *exec(const type_index & /*ti*/, void * in, void * /*copy*/,
      void * /*buf*/, bool /*fwd*/, size_t /*nthreads*/) const
      { return in; }
  };

template <typename Tfs> class rfftp2: public rfftpass<Tfs>
  {
  private:
    size_t l1, ido;
    static constexpr size_t ip=2;
    aligned_array<Tfs> wa;

    auto WA(size_t x, size_t i) const
      { return wa[i+x*(ido-1)]; }

    template<bool fwd, typename Tfd> Tfd *exec_ (Tfd * DUCC0_RESTRICT cc,
      Tfd * DUCC0_RESTRICT ch, Tfd * /*buf*/, size_t /*nthreads*/) const
      {
      if constexpr(fwd)
        {
        auto CC = [cc,this](size_t a, size_t b, size_t c) -> const Tfd&
          { return cc[a+ido*(b+l1*c)]; };
        auto CH = [ch,this](size_t a, size_t b, size_t c) -> Tfd&
          { return ch[a+ido*(b+ip*c)]; };
        for (size_t k=0; k<l1; k++)
          PM (CH(0,0,k),CH(ido-1,1,k),CC(0,k,0),CC(0,k,1));
        if ((ido&1)==0)
          for (size_t k=0; k<l1; k++)
            {
            CH(    0,1,k) = -CC(ido-1,k,1);
            CH(ido-1,0,k) =  CC(ido-1,k,0);
            }
        if (ido<=2) return ch;
        for (size_t k=0; k<l1; k++)
          for (size_t i=2; i<ido; i+=2)
            {
            size_t ic=ido-i;
            Tfd tr2, ti2;
            MULPM (tr2,ti2,WA(0,i-2),WA(0,i-1),CC(i-1,k,1),CC(i,k,1));
            PM (CH(i-1,0,k),CH(ic-1,1,k),CC(i-1,k,0),tr2);
            PM (CH(i  ,0,k),CH(ic  ,1,k),ti2,CC(i  ,k,0));
            }
        }
      else
        {
        auto CC = [cc,this](size_t a, size_t b, size_t c) -> const Tfd&
          { return cc[a+ido*(b+ip*c)]; };
        auto CH = [ch,this](size_t a, size_t b, size_t c) -> Tfd&
          { return ch[a+ido*(b+l1*c)]; };

        for (size_t k=0; k<l1; k++)
          PM (CH(0,k,0),CH(0,k,1),CC(0,0,k),CC(ido-1,1,k));
        if ((ido&1)==0)
          for (size_t k=0; k<l1; k++)
            {
            CH(ido-1,k,0) = Tfs( 2)*CC(ido-1,0,k);
            CH(ido-1,k,1) = Tfs(-2)*CC(0    ,1,k);
            }
        if (ido<=2) return ch;
        for (size_t k=0; k<l1;++k)
          for (size_t i=2; i<ido; i+=2)
            {
            size_t ic=ido-i;
            Tfd ti2, tr2;
            PM (CH(i-1,k,0),tr2,CC(i-1,0,k),CC(ic-1,1,k));
            PM (ti2,CH(i  ,k,0),CC(i  ,0,k),CC(ic  ,1,k));
            MULPM (CH(i,k,1),CH(i-1,k,1),WA(0,i-2),WA(0,i-1),ti2,tr2);
            }
        }
      return ch;
      }

  public:
    rfftp2(size_t l1_, size_t ido_, const Troots<Tfs> &roots)
      : l1(l1_), ido(ido_), wa((ip-1)*(ido-1))
      {
      size_t N=ip*l1*ido;
      size_t rfct = roots->size()/N;
      MR_assert(roots->size()==N*rfct, "mismatch");
      for (size_t j=1; j<ip; ++j)
        for (size_t i=1; i<=(ido-1)/2; ++i)
          {
          auto val = (*roots)[rfct*j*l1*i];
          wa[(j-1)*(ido-1)+2*i-2] = val.r;
          wa[(j-1)*(ido-1)+2*i-1] = val.i;
          }
      }

    virtual size_t bufsize() const { return 0; }
    virtual bool needs_copy() const { return true; }

    POCKETFFT_EXEC_DISPATCH
  };
// a2=a+b; b2=i*(b-a);
#define POCKETFFT_REARRANGE(rx, ix, ry, iy) \
  {\
  auto t1=rx+ry, t2=ry-rx, t3=ix+iy, t4=ix-iy; \
  rx=t1; ix=t3; ry=t4; iy=t2; \
  }

template <typename Tfs> class rfftp3: public rfftpass<Tfs>
  {
  private:
    size_t l1, ido;
    static constexpr size_t ip=3;
    aligned_array<Tfs> wa;

    auto WA(size_t x, size_t i) const
      { return wa[i+x*(ido-1)]; }

    template<bool fwd, typename Tfd> Tfd *exec_ (Tfd * DUCC0_RESTRICT cc,
      Tfd * DUCC0_RESTRICT ch, Tfd * /*buf*/, size_t /*nthreads*/) const
      {
      constexpr Tfs taur=Tfs(-0.5),
                    taui=Tfs(0.8660254037844386467637231707529362L);
      if constexpr(fwd)
        {
        auto CC = [cc,this](size_t a, size_t b, size_t c) -> const Tfd&
          { return cc[a+ido*(b+l1*c)]; };
        auto CH = [ch,this](size_t a, size_t b, size_t c) -> Tfd&
          { return ch[a+ido*(b+ip*c)]; };
        for (size_t k=0; k<l1; k++)
          {
          Tfd cr2=CC(0,k,1)+CC(0,k,2);
          CH(0,0,k) = CC(0,k,0)+cr2;
          CH(0,2,k) = taui*(CC(0,k,2)-CC(0,k,1));
          CH(ido-1,1,k) = CC(0,k,0)+taur*cr2;
          }
        if (ido==1) return ch;
        for (size_t k=0; k<l1; k++)
          for (size_t i=2; i<ido; i+=2)
            {
            size_t ic=ido-i;
            Tfd di2, di3, dr2, dr3;
            MULPM (dr2,di2,WA(0,i-2),WA(0,i-1),CC(i-1,k,1),CC(i,k,1)); // d2=conj(WA0)*CC1
            MULPM (dr3,di3,WA(1,i-2),WA(1,i-1),CC(i-1,k,2),CC(i,k,2)); // d3=conj(WA1)*CC2
            POCKETFFT_REARRANGE(dr2, di2, dr3, di3);
            CH(i-1,0,k) = CC(i-1,k,0)+dr2; // c add
            CH(i  ,0,k) = CC(i  ,k,0)+di2;
            Tfd tr2 = CC(i-1,k,0)+taur*dr2; // c add
            Tfd ti2 = CC(i  ,k,0)+taur*di2;
            Tfd tr3 = taui*dr3;  // t3 = taui*i*(d3-d2)?
            Tfd ti3 = taui*di3;
            PM(CH(i-1,2,k),CH(ic-1,1,k),tr2,tr3); // PM(i) = t2+t3
            PM(CH(i  ,2,k),CH(ic  ,1,k),ti3,ti2); // PM(ic) = conj(t2-t3)
            }
        }
      else
        {
        auto CC = [cc,this](size_t a, size_t b, size_t c) -> const Tfd&
          { return cc[a+ido*(b+ip*c)]; };
        auto CH = [ch,this](size_t a, size_t b, size_t c) -> Tfd&
          { return ch[a+ido*(b+l1*c)]; };

        for (size_t k=0; k<l1; k++)
          {
          Tfd tr2=Tfs(2)*CC(ido-1,1,k);
          Tfd cr2=CC(0,0,k)+taur*tr2;
          CH(0,k,0)=CC(0,0,k)+tr2;
          Tfd ci3=Tfs(2)*taui*CC(0,2,k);
          PM (CH(0,k,2),CH(0,k,1),cr2,ci3);
          }
        if (ido==1) return ch;
        for (size_t k=0; k<l1; k++)
          for (size_t i=2, ic=ido-2; i<ido; i+=2, ic-=2)
            {
            Tfd tr2=CC(i-1,2,k)+CC(ic-1,1,k); // t2=CC(I) + conj(CC(ic))
            Tfd ti2=CC(i  ,2,k)-CC(ic  ,1,k);
            Tfd cr2=CC(i-1,0,k)+taur*tr2;     // c2=CC +taur*t2
            Tfd ci2=CC(i  ,0,k)+taur*ti2;
            CH(i-1,k,0)=CC(i-1,0,k)+tr2;         // CH=CC+t2
            CH(i  ,k,0)=CC(i  ,0,k)+ti2;
            Tfd cr3=taui*(CC(i-1,2,k)-CC(ic-1,1,k));// c3=taui*(CC(i)-conj(CC(ic)))
            Tfd ci3=taui*(CC(i  ,2,k)+CC(ic  ,1,k));
            Tfd di2, di3, dr2, dr3;
            PM(dr3,dr2,cr2,ci3); // d2= (cr2-ci3, ci2+cr3) = c2+i*c3
            PM(di2,di3,ci2,cr3); // d3= (cr2+ci3, ci2-cr3) = c2-i*c3
            MULPM(CH(i,k,1),CH(i-1,k,1),WA(0,i-2),WA(0,i-1),di2,dr2); // ch = WA*d2
            MULPM(CH(i,k,2),CH(i-1,k,2),WA(1,i-2),WA(1,i-1),di3,dr3);
            }
        }
      return ch;
      }

  public:
    rfftp3(size_t l1_, size_t ido_, const Troots<Tfs> &roots)
      : l1(l1_), ido(ido_), wa((ip-1)*(ido-1))
      {
      MR_assert(ido&1, "ido must be odd");
      size_t N=ip*l1*ido;
      size_t rfct = roots->size()/N;
      MR_assert(roots->size()==N*rfct, "mismatch");
      for (size_t j=1; j<ip; ++j)
        for (size_t i=1; i<=(ido-1)/2; ++i)
          {
          auto val = (*roots)[rfct*j*l1*i];
          wa[(j-1)*(ido-1)+2*i-2] = val.r;
          wa[(j-1)*(ido-1)+2*i-1] = val.i;
          }
      }

    virtual size_t bufsize() const { return 0; }
    virtual bool needs_copy() const { return true; }

    POCKETFFT_EXEC_DISPATCH
  };

template <typename Tfs> class rfftp4: public rfftpass<Tfs>
  {
  private:
    size_t l1, ido;
    static constexpr size_t ip=4;
    aligned_array<Tfs> wa;

    auto WA(size_t x, size_t i) const
      { return wa[i+x*(ido-1)]; }

    template<bool fwd, typename Tfd> Tfd *exec_ (Tfd * DUCC0_RESTRICT cc,
      Tfd * DUCC0_RESTRICT ch, Tfd * /*buf*/, size_t /*nthreads*/) const
      {
      if constexpr(fwd)
        {
        constexpr Tfs hsqt2=Tfs(0.707106781186547524400844362104849L);
        auto CC = [cc,this](size_t a, size_t b, size_t c) -> const Tfd&
          { return cc[a+ido*(b+l1*c)]; };
        auto CH = [ch,this](size_t a, size_t b, size_t c) -> Tfd&
          { return ch[a+ido*(b+ip*c)]; };

        for (size_t k=0; k<l1; k++)
          {
          Tfd tr1,tr2;
          PM (tr1,CH(0,2,k),CC(0,k,3),CC(0,k,1));
          PM (tr2,CH(ido-1,1,k),CC(0,k,0),CC(0,k,2));
          PM (CH(0,0,k),CH(ido-1,3,k),tr2,tr1);
          }
        if ((ido&1)==0)
          for (size_t k=0; k<l1; k++)
            {
            Tfd ti1=-hsqt2*(CC(ido-1,k,1)+CC(ido-1,k,3));
            Tfd tr1= hsqt2*(CC(ido-1,k,1)-CC(ido-1,k,3));
            PM (CH(ido-1,0,k),CH(ido-1,2,k),CC(ido-1,k,0),tr1);
            PM (CH(    0,3,k),CH(    0,1,k),ti1,CC(ido-1,k,2));
            }
        if (ido<=2) return ch;
        for (size_t k=0; k<l1; k++)
          for (size_t i=2; i<ido; i+=2)
            {
            size_t ic=ido-i;
            Tfd ci2, ci3, ci4, cr2, cr3, cr4, ti1, ti2, ti3, ti4, tr1, tr2, tr3, tr4;
            MULPM(cr2,ci2,WA(0,i-2),WA(0,i-1),CC(i-1,k,1),CC(i,k,1));
            MULPM(cr3,ci3,WA(1,i-2),WA(1,i-1),CC(i-1,k,2),CC(i,k,2));
            MULPM(cr4,ci4,WA(2,i-2),WA(2,i-1),CC(i-1,k,3),CC(i,k,3));
            PM(tr1,tr4,cr4,cr2);
            PM(ti1,ti4,ci2,ci4);
            PM(tr2,tr3,CC(i-1,k,0),cr3);
            PM(ti2,ti3,CC(i  ,k,0),ci3);
            PM(CH(i-1,0,k),CH(ic-1,3,k),tr2,tr1);
            PM(CH(i  ,0,k),CH(ic  ,3,k),ti1,ti2);
            PM(CH(i-1,2,k),CH(ic-1,1,k),tr3,ti4);
            PM(CH(i  ,2,k),CH(ic  ,1,k),tr4,ti3);
            }
        }
      else
        {
        constexpr Tfs sqrt2=Tfs(1.414213562373095048801688724209698L);
        auto CC = [cc,this](size_t a, size_t b, size_t c) -> const Tfd&
          { return cc[a+ido*(b+ip*c)]; };
        auto CH = [ch,this](size_t a, size_t b, size_t c) -> Tfd&
          { return ch[a+ido*(b+l1*c)]; };

        for (size_t k=0; k<l1; k++)
          {
          Tfd tr1, tr2;
          PM (tr2,tr1,CC(0,0,k),CC(ido-1,3,k));
          Tfd tr3=Tfs(2)*CC(ido-1,1,k);
          Tfd tr4=Tfs(2)*CC(0,2,k);
          PM (CH(0,k,0),CH(0,k,2),tr2,tr3);
          PM (CH(0,k,3),CH(0,k,1),tr1,tr4);
          }
        if ((ido&1)==0)
          for (size_t k=0; k<l1; k++)
            {
            Tfd tr1,tr2,ti1,ti2;
            PM (ti1,ti2,CC(0    ,3,k),CC(0    ,1,k));
            PM (tr2,tr1,CC(ido-1,0,k),CC(ido-1,2,k));
            CH(ido-1,k,0)=tr2+tr2;
            CH(ido-1,k,1)=sqrt2*(tr1-ti1);
            CH(ido-1,k,2)=ti2+ti2;
            CH(ido-1,k,3)=-sqrt2*(tr1+ti1);
            }
        if (ido<=2) return ch;
        for (size_t k=0; k<l1;++k)
          for (size_t i=2; i<ido; i+=2)
            {
            Tfd ci2, ci3, ci4, cr2, cr3, cr4, ti1, ti2, ti3, ti4, tr1, tr2, tr3, tr4;
            size_t ic=ido-i;
            PM (tr2,tr1,CC(i-1,0,k),CC(ic-1,3,k));
            PM (ti1,ti2,CC(i  ,0,k),CC(ic  ,3,k));
            PM (tr4,ti3,CC(i  ,2,k),CC(ic  ,1,k));
            PM (tr3,ti4,CC(i-1,2,k),CC(ic-1,1,k));
            PM (CH(i-1,k,0),cr3,tr2,tr3);
            PM (CH(i  ,k,0),ci3,ti2,ti3);
            PM (cr4,cr2,tr1,tr4);
            PM (ci2,ci4,ti1,ti4);
            MULPM (CH(i,k,1),CH(i-1,k,1),WA(0,i-2),WA(0,i-1),ci2,cr2);
            MULPM (CH(i,k,2),CH(i-1,k,2),WA(1,i-2),WA(1,i-1),ci3,cr3);
            MULPM (CH(i,k,3),CH(i-1,k,3),WA(2,i-2),WA(2,i-1),ci4,cr4);
            }
        }
      return ch;
      }

  public:
    rfftp4(size_t l1_, size_t ido_, const Troots<Tfs> &roots)
      : l1(l1_), ido(ido_), wa((ip-1)*(ido-1))
      {
      size_t N=ip*l1*ido;
      size_t rfct = roots->size()/N;
      MR_assert(roots->size()==N*rfct, "mismatch");
      for (size_t j=1; j<ip; ++j)
        for (size_t i=1; i<=(ido-1)/2; ++i)
          {
          auto val = (*roots)[rfct*j*l1*i];
          wa[(j-1)*(ido-1)+2*i-2] = val.r;
          wa[(j-1)*(ido-1)+2*i-1] = val.i;
          }
      }

    virtual size_t bufsize() const { return 0; }
    virtual bool needs_copy() const { return true; }

    POCKETFFT_EXEC_DISPATCH
  };

template <typename Tfs> class rfftp5: public rfftpass<Tfs>
  {
  private:
    size_t l1, ido;
    static constexpr size_t ip=5;
    aligned_array<Tfs> wa;

    auto WA(size_t x, size_t i) const
      { return wa[i+x*(ido-1)]; }

    template<bool fwd, typename Tfd> Tfd *exec_ (Tfd * DUCC0_RESTRICT cc,
      Tfd * DUCC0_RESTRICT ch, Tfd * /*buf*/, size_t /*nthreads*/) const
      {
      constexpr Tfs tr11= Tfs(0.3090169943749474241022934171828191L),
                    ti11= Tfs(0.9510565162951535721164393333793821L),
                    tr12= Tfs(-0.8090169943749474241022934171828191L),
                    ti12= Tfs(0.5877852522924731291687059546390728L);

      if constexpr(fwd)
        {
        auto CC = [cc,this](size_t a, size_t b, size_t c) -> const Tfd&
          { return cc[a+ido*(b+l1*c)]; };
        auto CH = [ch,this](size_t a, size_t b, size_t c) -> Tfd&
          { return ch[a+ido*(b+ip*c)]; };

        for (size_t k=0; k<l1; k++)
          {
          Tfd cr2, cr3, ci4, ci5;
          PM (cr2,ci5,CC(0,k,4),CC(0,k,1));
          PM (cr3,ci4,CC(0,k,3),CC(0,k,2));
          CH(0,0,k)=CC(0,k,0)+cr2+cr3;
          CH(ido-1,1,k)=CC(0,k,0)+tr11*cr2+tr12*cr3;
          CH(0,2,k)=ti11*ci5+ti12*ci4;
          CH(ido-1,3,k)=CC(0,k,0)+tr12*cr2+tr11*cr3;
          CH(0,4,k)=ti12*ci5-ti11*ci4;
          }
        if (ido==1) return ch;
        for (size_t k=0; k<l1;++k)
          for (size_t i=2, ic=ido-2; i<ido; i+=2, ic-=2)
            {
            Tfd di2, di3, di4, di5, dr2, dr3, dr4, dr5;
            MULPM (dr2,di2,WA(0,i-2),WA(0,i-1),CC(i-1,k,1),CC(i,k,1));
            MULPM (dr3,di3,WA(1,i-2),WA(1,i-1),CC(i-1,k,2),CC(i,k,2));
            MULPM (dr4,di4,WA(2,i-2),WA(2,i-1),CC(i-1,k,3),CC(i,k,3));
            MULPM (dr5,di5,WA(3,i-2),WA(3,i-1),CC(i-1,k,4),CC(i,k,4));
            POCKETFFT_REARRANGE(dr2, di2, dr5, di5);
            POCKETFFT_REARRANGE(dr3, di3, dr4, di4);
            CH(i-1,0,k)=CC(i-1,k,0)+dr2+dr3;
            CH(i  ,0,k)=CC(i  ,k,0)+di2+di3;
            Tfd tr2=CC(i-1,k,0)+tr11*dr2+tr12*dr3;
            Tfd ti2=CC(i  ,k,0)+tr11*di2+tr12*di3;
            Tfd tr3=CC(i-1,k,0)+tr12*dr2+tr11*dr3;
            Tfd ti3=CC(i  ,k,0)+tr12*di2+tr11*di3;
            Tfd tr5 = ti11*dr5 + ti12*dr4;
            Tfd ti5 = ti11*di5 + ti12*di4;
            Tfd tr4 = ti12*dr5 - ti11*dr4;
            Tfd ti4 = ti12*di5 - ti11*di4;
            PM(CH(i-1,2,k),CH(ic-1,1,k),tr2,tr5);
            PM(CH(i  ,2,k),CH(ic  ,1,k),ti5,ti2);
            PM(CH(i-1,4,k),CH(ic-1,3,k),tr3,tr4);
            PM(CH(i  ,4,k),CH(ic  ,3,k),ti4,ti3);
            }
        }
      else
        {
        auto CC = [cc,this](size_t a, size_t b, size_t c) -> const Tfd&
          { return cc[a+ido*(b+ip*c)]; };
        auto CH = [ch,this](size_t a, size_t b, size_t c) -> Tfd&
          { return ch[a+ido*(b+l1*c)]; };

        for (size_t k=0; k<l1; k++)
          {
          Tfd ti5=CC(0,2,k)+CC(0,2,k);
          Tfd ti4=CC(0,4,k)+CC(0,4,k);
          Tfd tr2=CC(ido-1,1,k)+CC(ido-1,1,k);
          Tfd tr3=CC(ido-1,3,k)+CC(ido-1,3,k);
          CH(0,k,0)=CC(0,0,k)+tr2+tr3;
          Tfd cr2=CC(0,0,k)+tr11*tr2+tr12*tr3;
          Tfd cr3=CC(0,0,k)+tr12*tr2+tr11*tr3;
          Tfd ci4, ci5;
          MULPM(ci5,ci4,ti5,ti4,ti11,ti12);
          PM(CH(0,k,4),CH(0,k,1),cr2,ci5);
          PM(CH(0,k,3),CH(0,k,2),cr3,ci4);
          }
        if (ido==1) return ch;
        for (size_t k=0; k<l1;++k)
          for (size_t i=2, ic=ido-2; i<ido; i+=2, ic-=2)
            {
            Tfd tr2, tr3, tr4, tr5, ti2, ti3, ti4, ti5;
            PM(tr2,tr5,CC(i-1,2,k),CC(ic-1,1,k));
            PM(ti5,ti2,CC(i  ,2,k),CC(ic  ,1,k));
            PM(tr3,tr4,CC(i-1,4,k),CC(ic-1,3,k));
            PM(ti4,ti3,CC(i  ,4,k),CC(ic  ,3,k));
            CH(i-1,k,0)=CC(i-1,0,k)+tr2+tr3;
            CH(i  ,k,0)=CC(i  ,0,k)+ti2+ti3;
            Tfd cr2=CC(i-1,0,k)+tr11*tr2+tr12*tr3;
            Tfd ci2=CC(i  ,0,k)+tr11*ti2+tr12*ti3;
            Tfd cr3=CC(i-1,0,k)+tr12*tr2+tr11*tr3;
            Tfd ci3=CC(i  ,0,k)+tr12*ti2+tr11*ti3;
            Tfd ci4, ci5, cr5, cr4;
            MULPM(cr5,cr4,tr5,tr4,ti11,ti12);
            MULPM(ci5,ci4,ti5,ti4,ti11,ti12);
            Tfd dr2, dr3, dr4, dr5, di2, di3, di4, di5;
            PM(dr4,dr3,cr3,ci4);
            PM(di3,di4,ci3,cr4);
            PM(dr5,dr2,cr2,ci5);
            PM(di2,di5,ci2,cr5);
            MULPM(CH(i,k,1),CH(i-1,k,1),WA(0,i-2),WA(0,i-1),di2,dr2);
            MULPM(CH(i,k,2),CH(i-1,k,2),WA(1,i-2),WA(1,i-1),di3,dr3);
            MULPM(CH(i,k,3),CH(i-1,k,3),WA(2,i-2),WA(2,i-1),di4,dr4);
            MULPM(CH(i,k,4),CH(i-1,k,4),WA(3,i-2),WA(3,i-1),di5,dr5);
            }
        }
      return ch;
      }

  public:
    rfftp5(size_t l1_, size_t ido_, const Troots<Tfs> &roots)
      : l1(l1_), ido(ido_), wa((ip-1)*(ido-1))
      {
      MR_assert(ido&1, "ido must be odd");
      size_t N=ip*l1*ido;
      size_t rfct = roots->size()/N;
      MR_assert(roots->size()==N*rfct, "mismatch");
      for (size_t j=1; j<ip; ++j)
        for (size_t i=1; i<=(ido-1)/2; ++i)
          {
          auto val = (*roots)[rfct*j*l1*i];
          wa[(j-1)*(ido-1)+2*i-2] = val.r;
          wa[(j-1)*(ido-1)+2*i-1] = val.i;
          }
      }

    virtual size_t bufsize() const { return 0; }
    virtual bool needs_copy() const { return true; }

    POCKETFFT_EXEC_DISPATCH
  };

template <typename Tfs> class rfftpg: public rfftpass<Tfs>
  {
  private:
    size_t l1, ido;
    size_t ip;
    aligned_array<Tfs> wa, csarr;

    template<bool fwd, typename Tfd> Tfd *exec_ (Tfd * DUCC0_RESTRICT cc,
      Tfd * DUCC0_RESTRICT ch, Tfd * /*buf*/, size_t /*nthreads*/) const
      {
      if constexpr(fwd)
        {
        size_t ipph=(ip+1)/2;
        size_t idl1 = ido*l1;

        auto CC = [cc,this](size_t a, size_t b, size_t c) -> Tfd&
          { return cc[a+ido*(b+ip*c)]; };
        auto CH = [ch,this](size_t a, size_t b, size_t c) -> const Tfd&
          { return ch[a+ido*(b+l1*c)]; };
        auto C1 = [cc,this] (size_t a, size_t b, size_t c) -> Tfd&
          { return cc[a+ido*(b+l1*c)]; };
        auto C2 = [cc,idl1] (size_t a, size_t b) -> Tfd&
          { return cc[a+idl1*b]; };
        auto CH2 = [ch,idl1] (size_t a, size_t b) -> Tfd&
          { return ch[a+idl1*b]; };

        if (ido>1)
          {
          for (size_t j=1, jc=ip-1; j<ipph; ++j,--jc)              // 114
            {
            size_t is=(j-1)*(ido-1),
                   is2=(jc-1)*(ido-1);
            for (size_t k=0; k<l1; ++k)                            // 113
              {
              size_t idij=is;
              size_t idij2=is2;
              for (size_t i=1; i<=ido-2; i+=2)                      // 112
                {
                Tfd t1=C1(i,k,j ), t2=C1(i+1,k,j ),
                    t3=C1(i,k,jc), t4=C1(i+1,k,jc);
                Tfd x1=wa[idij]*t1 + wa[idij+1]*t2,
                    x2=wa[idij]*t2 - wa[idij+1]*t1,
                    x3=wa[idij2]*t3 + wa[idij2+1]*t4,
                    x4=wa[idij2]*t4 - wa[idij2+1]*t3;
                PM(C1(i,k,j),C1(i+1,k,jc),x3,x1);
                PM(C1(i+1,k,j),C1(i,k,jc),x2,x4);
                idij+=2;
                idij2+=2;
                }
              }
            }
          }

        for (size_t j=1, jc=ip-1; j<ipph; ++j,--jc)                // 123
          for (size_t k=0; k<l1; ++k)                              // 122
            MPINPLACE(C1(0,k,jc), C1(0,k,j));

      //everything in C
      //memset(ch,0,ip*l1*ido*sizeof(double));

        for (size_t l=1,lc=ip-1; l<ipph; ++l,--lc)                 // 127
          {
          for (size_t ik=0; ik<idl1; ++ik)                         // 124
            {
            CH2(ik,l ) = C2(ik,0)+csarr[2*l]*C2(ik,1)+csarr[4*l]*C2(ik,2);
            CH2(ik,lc) = csarr[2*l+1]*C2(ik,ip-1)+csarr[4*l+1]*C2(ik,ip-2);
            }
          size_t iang = 2*l;
          size_t j=3, jc=ip-3;
          for (; j<ipph-3; j+=4,jc-=4)              // 126
            {
            iang+=l; if (iang>=ip) iang-=ip;
            Tfs ar1=csarr[2*iang], ai1=csarr[2*iang+1];
            iang+=l; if (iang>=ip) iang-=ip;
            Tfs ar2=csarr[2*iang], ai2=csarr[2*iang+1];
            iang+=l; if (iang>=ip) iang-=ip;
            Tfs ar3=csarr[2*iang], ai3=csarr[2*iang+1];
            iang+=l; if (iang>=ip) iang-=ip;
            Tfs ar4=csarr[2*iang], ai4=csarr[2*iang+1];
            for (size_t ik=0; ik<idl1; ++ik)                       // 125
              {
              CH2(ik,l ) += ar1*C2(ik,j )+ar2*C2(ik,j +1)
                           +ar3*C2(ik,j +2)+ar4*C2(ik,j +3);
              CH2(ik,lc) += ai1*C2(ik,jc)+ai2*C2(ik,jc-1)
                           +ai3*C2(ik,jc-2)+ai4*C2(ik,jc-3);
              }
            }
          for (; j<ipph-1; j+=2,jc-=2)              // 126
            {
            iang+=l; if (iang>=ip) iang-=ip;
            Tfs ar1=csarr[2*iang], ai1=csarr[2*iang+1];
            iang+=l; if (iang>=ip) iang-=ip;
            Tfs ar2=csarr[2*iang], ai2=csarr[2*iang+1];
            for (size_t ik=0; ik<idl1; ++ik)                       // 125
              {
              CH2(ik,l ) += ar1*C2(ik,j )+ar2*C2(ik,j +1);
              CH2(ik,lc) += ai1*C2(ik,jc)+ai2*C2(ik,jc-1);
              }
            }
          for (; j<ipph; ++j,--jc)              // 126
            {
            iang+=l; if (iang>=ip) iang-=ip;
            Tfs ar=csarr[2*iang], ai=csarr[2*iang+1];
            for (size_t ik=0; ik<idl1; ++ik)                       // 125
              {
              CH2(ik,l ) += ar*C2(ik,j );
              CH2(ik,lc) += ai*C2(ik,jc);
              }
            }
          }
        for (size_t ik=0; ik<idl1; ++ik)                         // 101
          CH2(ik,0) = C2(ik,0);
        for (size_t j=1; j<ipph; ++j)                              // 129
          for (size_t ik=0; ik<idl1; ++ik)                         // 128
            CH2(ik,0) += C2(ik,j);

      // everything in CH at this point!
      //memset(cc,0,ip*l1*ido*sizeof(double));

        for (size_t k=0; k<l1; ++k)                                // 131
          for (size_t i=0; i<ido; ++i)                             // 130
            CC(i,0,k) = CH(i,k,0);

        for (size_t j=1, jc=ip-1; j<ipph; ++j,--jc)                // 137
          {
          size_t j2=2*j-1;
          for (size_t k=0; k<l1; ++k)                              // 136
            {
            CC(ido-1,j2,k) = CH(0,k,j);
            CC(0,j2+1,k) = CH(0,k,jc);
            }
          }

        if (ido==1) return cc;

        for (size_t j=1, jc=ip-1; j<ipph; ++j,--jc)                // 140
          {
          size_t j2=2*j-1;
          for(size_t k=0; k<l1; ++k)                               // 139
            for(size_t i=1, ic=ido-i-2; i<=ido-2; i+=2, ic-=2)      // 138
              {
              CC(i   ,j2+1,k) = CH(i  ,k,j )+CH(i  ,k,jc);
              CC(ic  ,j2  ,k) = CH(i  ,k,j )-CH(i  ,k,jc);
              CC(i+1 ,j2+1,k) = CH(i+1,k,j )+CH(i+1,k,jc);
              CC(ic+1,j2  ,k) = CH(i+1,k,jc)-CH(i+1,k,j );
              }
          }
        return cc;
        }
      else
        {
        size_t ipph=(ip+1)/ 2;
        size_t idl1 = ido*l1;

        auto CC = [cc,this](size_t a, size_t b, size_t c) -> const Tfd&
          { return cc[a+ido*(b+ip*c)]; };
        auto CH = [ch,this](size_t a, size_t b, size_t c) -> Tfd&
          { return ch[a+ido*(b+l1*c)]; };
        auto C1 = [cc,this](size_t a, size_t b, size_t c) -> const Tfd&
          { return cc[a+ido*(b+l1*c)]; };
        auto C2 = [cc,idl1](size_t a, size_t b) -> Tfd&
          { return cc[a+idl1*b]; };
        auto CH2 = [ch,idl1](size_t a, size_t b) -> Tfd&
          { return ch[a+idl1*b]; };

        for (size_t k=0; k<l1; ++k)        // 102
          for (size_t i=0; i<ido; ++i)     // 101
            CH(i,k,0) = CC(i,0,k);
        for (size_t j=1, jc=ip-1; j<ipph; ++j, --jc)   // 108
          {
          size_t j2=2*j-1;
          for (size_t k=0; k<l1; ++k)
            {
            CH(0,k,j ) = Tfs(2)*CC(ido-1,j2,k);
            CH(0,k,jc) = Tfs(2)*CC(0,j2+1,k);
            }
          }

        if (ido!=1)
          {
          for (size_t j=1, jc=ip-1; j<ipph; ++j,--jc)   // 111
            {
            size_t j2=2*j-1;
            for (size_t k=0; k<l1; ++k)
              for (size_t i=1, ic=ido-i-2; i<=ido-2; i+=2, ic-=2)      // 109
                {
                CH(i  ,k,j ) = CC(i  ,j2+1,k)+CC(ic  ,j2,k);
                CH(i  ,k,jc) = CC(i  ,j2+1,k)-CC(ic  ,j2,k);
                CH(i+1,k,j ) = CC(i+1,j2+1,k)-CC(ic+1,j2,k);
                CH(i+1,k,jc) = CC(i+1,j2+1,k)+CC(ic+1,j2,k);
                }
            }
          }
        for (size_t l=1,lc=ip-1; l<ipph; ++l,--lc)
          {
          for (size_t ik=0; ik<idl1; ++ik)
            {
            C2(ik,l ) = CH2(ik,0)+csarr[2*l]*CH2(ik,1)+csarr[4*l]*CH2(ik,2);
            C2(ik,lc) = csarr[2*l+1]*CH2(ik,ip-1)+csarr[4*l+1]*CH2(ik,ip-2);
            }
          size_t iang=2*l;
          size_t j=3,jc=ip-3;
          for(; j<ipph-3; j+=4,jc-=4)
            {
            iang+=l; if(iang>ip) iang-=ip;
            Tfs ar1=csarr[2*iang], ai1=csarr[2*iang+1];
            iang+=l; if(iang>ip) iang-=ip;
            Tfs ar2=csarr[2*iang], ai2=csarr[2*iang+1];
            iang+=l; if(iang>ip) iang-=ip;
            Tfs ar3=csarr[2*iang], ai3=csarr[2*iang+1];
            iang+=l; if(iang>ip) iang-=ip;
            Tfs ar4=csarr[2*iang], ai4=csarr[2*iang+1];
            for (size_t ik=0; ik<idl1; ++ik)
              {
              C2(ik,l ) += ar1*CH2(ik,j )+ar2*CH2(ik,j +1)
                          +ar3*CH2(ik,j +2)+ar4*CH2(ik,j +3);
              C2(ik,lc) += ai1*CH2(ik,jc)+ai2*CH2(ik,jc-1)
                          +ai3*CH2(ik,jc-2)+ai4*CH2(ik,jc-3);
              }
            }
          for(; j<ipph-1; j+=2,jc-=2)
            {
            iang+=l; if(iang>ip) iang-=ip;
            Tfs ar1=csarr[2*iang], ai1=csarr[2*iang+1];
            iang+=l; if(iang>ip) iang-=ip;
            Tfs ar2=csarr[2*iang], ai2=csarr[2*iang+1];
            for (size_t ik=0; ik<idl1; ++ik)
              {
              C2(ik,l ) += ar1*CH2(ik,j )+ar2*CH2(ik,j +1);
              C2(ik,lc) += ai1*CH2(ik,jc)+ai2*CH2(ik,jc-1);
              }
            }
          for(; j<ipph; ++j,--jc)
            {
            iang+=l; if(iang>ip) iang-=ip;
            Tfs war=csarr[2*iang], wai=csarr[2*iang+1];
            for (size_t ik=0; ik<idl1; ++ik)
              {
              C2(ik,l ) += war*CH2(ik,j );
              C2(ik,lc) += wai*CH2(ik,jc);
              }
            }
          }
        for (size_t j=1; j<ipph; ++j)
          for (size_t ik=0; ik<idl1; ++ik)
            CH2(ik,0) += CH2(ik,j);
        for (size_t j=1, jc=ip-1; j<ipph; ++j,--jc)   // 124
          for (size_t k=0; k<l1; ++k)
            PM(CH(0,k,jc),CH(0,k,j),C1(0,k,j),C1(0,k,jc));

        if (ido==1) return ch;

        for (size_t j=1, jc=ip-1; j<ipph; ++j, --jc)  // 127
          for (size_t k=0; k<l1; ++k)
            for (size_t i=1; i<=ido-2; i+=2)
              {
              CH(i  ,k,j ) = C1(i  ,k,j)-C1(i+1,k,jc);
              CH(i  ,k,jc) = C1(i  ,k,j)+C1(i+1,k,jc);
              CH(i+1,k,j ) = C1(i+1,k,j)+C1(i  ,k,jc);
              CH(i+1,k,jc) = C1(i+1,k,j)-C1(i  ,k,jc);
              }

      // All in CH

        for (size_t j=1; j<ip; ++j)
          {
          size_t is = (j-1)*(ido-1);
          for (size_t k=0; k<l1; ++k)
            {
            size_t idij = is;
            for (size_t i=1; i<=ido-2; i+=2)
              {
              Tfd t1=CH(i,k,j), t2=CH(i+1,k,j);
              CH(i  ,k,j) = wa[idij]*t1-wa[idij+1]*t2;
              CH(i+1,k,j) = wa[idij]*t2+wa[idij+1]*t1;
              idij+=2;
              }
            }
          }
        return ch;
        }
      }

  public:
    rfftpg(size_t l1_, size_t ido_, size_t ip_, const Troots<Tfs> &roots)
      : l1(l1_), ido(ido_), ip(ip_), wa((ip-1)*(ido-1)), csarr(2*ip)
      {
      MR_assert(ido&1, "ido must be odd");
      size_t N=ip*l1*ido;
      size_t rfct = roots->size()/N;
      MR_assert(roots->size()==N*rfct, "mismatch");
      for (size_t j=1; j<ip; ++j)
        for (size_t i=1; i<=(ido-1)/2; ++i)
          {
          auto val = (*roots)[rfct*j*l1*i];
          wa[(j-1)*(ido-1)+2*i-2] = val.r;
          wa[(j-1)*(ido-1)+2*i-1] = val.i;
          }
      csarr[0] = Tfs(1);
      csarr[1] = Tfs(0);
      for (size_t i=2, ic=2*ip-2; i<=ic; i+=2, ic-=2)
        {
        auto val = (*roots)[i/2*rfct*(N/ip)];
        csarr[i   ] = val.r;
        csarr[i +1] = val.i;
        csarr[ic  ] = val.r;
        csarr[ic+1] = -val.i;
        }
      }

    virtual size_t bufsize() const { return 0; }
    virtual bool needs_copy() const { return true; }

    POCKETFFT_EXEC_DISPATCH
  };

template <typename Tfs> class rfftpblue: public rfftpass<Tfs>
  {
  private:
    const size_t l1, ido, ip;
    aligned_array<Tfs> wa;
    const Tcpass<Tfs> cplan;
    size_t bufsz;
    bool need_cpy;

    auto WA(size_t x, size_t i) const
      { return wa[i+x*(ido-1)]; }

    template<bool fwd, typename Tfd> Tfd *exec_
      (Tfd * DUCC0_RESTRICT cc, Tfd * DUCC0_RESTRICT ch,
       Tfd * DUCC0_RESTRICT buf_, size_t nthreads) const
      {
      using Tcd = Cmplx<Tfd>;
      auto buf = reinterpret_cast<Tcd *>(buf_);
      Tcd *cc2 = &buf[0];
      Tcd *ch2 = &buf[ip];
      Tcd *subbuf = &buf[2*ip];
      static const auto ticd = tidx<Tcd *>();

      if constexpr(fwd)
        {
        auto CC = [cc,this](size_t a, size_t b, size_t c) -> const Tfd&
          { return cc[a+ido*(b+l1*c)]; };
        auto CH = [ch,this](size_t a, size_t b, size_t c) -> Tfd&
          { return ch[a+ido*(b+ip*c)]; };

        for (size_t k=0; k<l1; ++k)
          {
          // copy in
          for (size_t m=0; m<ip; ++m)
            cc2[m] = {CC(0,k,m),Tfd(0)};
          auto res = static_cast<Tcd *>(cplan->exec(ticd, cc2, ch2,
            subbuf, fwd, nthreads));
          // copy out
          CH(0,0,k) = res[0].r;
          for (size_t m=1; m<=ip/2; ++m)
            {
            CH(ido-1,2*m-1,k)=res[m].r;
            CH(0,2*m,k)=res[m].i;
            }
          }
        if (ido==1) return ch;
        size_t ipph = (ip+1)/2;
        for (size_t k=0; k<l1; ++k)
          for (size_t i=2, ic=ido-2; i<ido; i+=2, ic-=2)
            {
            // copy in
            cc2[0] = {CC(i-1,k,0),CC(i,k,0)};
            for (size_t m=1; m<ipph; ++m)
              {
              MULPM (cc2[m].r,cc2[m].i,WA(m-1,i-2),WA(m-1,i-1),CC(i-1,k,m),CC(i,k,m));
              MULPM (cc2[ip-m].r,cc2[ip-m].i,WA(ip-m-1,i-2),WA(ip-m-1,i-1),CC(i-1,k,ip-m),CC(i,k,ip-m));
              }
            auto res = static_cast<Tcd *>(cplan->exec(ticd, cc2, ch2,
              subbuf, fwd, nthreads));
            CH(i-1,0,k) = res[0].r;
            CH(i,0,k) = res[0].i;
            for (size_t m=1; m<ipph; ++m)
              {
              CH(i-1,2*m,k) = res[m].r;
              CH(ic-1,2*m-1,k) = res[ip-m].r;
              CH(i  ,2*m,k) = res[m].i;
              CH(ic  ,2*m-1,k) = -res[ip-m].i;
              }
            }
        }
      else
        {
        auto CC = [cc,this](size_t a, size_t b, size_t c) -> Tfd&
          { return cc[a+ido*(b+ip*c)]; };
        auto CH = [ch,this](size_t a, size_t b, size_t c) -> Tfd&
          { return ch[a+ido*(b+l1*c)]; };

        for (size_t k=0; k<l1; k++)
          {
          cc2[0] = {CC(0,0,k), Tfd(0)};
          for (size_t m=1; m<=ip/2; ++m)
            {
            cc2[m] = {CC(ido-1,2*m-1,k),CC(0,2*m,k)};
            cc2[ip-m] = {CC(ido-1,2*m-1,k),-CC(0,2*m,k)};
            }
          auto res = static_cast<Tcd *>(cplan->exec(ticd, cc2, ch2,
            subbuf, fwd, nthreads));
          for (size_t m=0; m<ip; ++m)
            CH(0,k,m) = res[m].r;
          }
        if (ido==1) return ch;
        for (size_t k=0; k<l1; ++k)
          for (size_t i=2, ic=ido-2; i<ido; i+=2, ic-=2)
            {
            // copy in
            cc2[0] = {CC(i-1,0,k),CC(i,0,k)};
            for (size_t m=1; m<=ip/2; ++m)
              {
              cc2[m] = {CC(i-1,2*m,k),CC(i,2*m,k)};
              cc2[ip-m] = {CC(ic-1,2*m-1,k),-CC(ic,2*m-1,k)};
              }
            auto res = static_cast<Tcd *>(cplan->exec(ticd, cc2, ch2,
              subbuf, fwd, nthreads));
            CH(i-1,k,0) = res[0].r;
            CH(i,k,0) = res[0].i;
            for (size_t m=1; m<ip; ++m)
              {
              MULPM(CH(i-1,k,m),CH(i,k,m),WA(m-1,i-2),-WA(m-1,i-1),res[m].r,res[m].i);
              MULPM(CH(i-1,k,ip-m),CH(i,k,ip-m),WA(ip-m-1,i-2),-WA(ip-m-1,i-1),res[ip-m].r,res[ip-m].i);
              }
            }
        }
      return ch;
      }

  public:
    rfftpblue(size_t l1_, size_t ido_, size_t ip_, const Troots<Tfs> &roots, bool vectorize=false)
      : l1(l1_), ido(ido_), ip(ip_), wa((ip-1)*(ido-1)),
        cplan(cfftpass<Tfs>::make_pass(1,1,ip,roots,vectorize))
      {
      MR_assert(ip&1, "Bluestein length must be odd");
      MR_assert(ido&1, "ido must be odd");
      size_t N=ip*l1*ido;
      auto rfct = roots->size()/N;
      MR_assert(roots->size()==N*rfct, "mismatch");
      for (size_t j=1; j<ip; ++j)
        for (size_t i=1; i<=(ido-1)/2; ++i)
          {
          auto val = (*roots)[rfct*j*l1*i];
          wa[(j-1)*(ido-1)+2*i-2] = val.r;
          wa[(j-1)*(ido-1)+2*i-1] = val.i;
          }
      }

    virtual size_t bufsize() const { return 4*ip + 2*cplan->bufsize(); }
    virtual bool needs_copy() const { return true; }

    POCKETFFT_EXEC_DISPATCH
  };

template <typename Tfs> class rfft_multipass: public rfftpass<Tfs>
  {
  private:
    const size_t l1, ido;
    size_t ip;
    vector<Trpass<Tfs>> passes;
    size_t bufsz;
    bool need_cpy;
    aligned_array<Tfs> wa;

    auto WA(size_t x, size_t i) const
      { return wa[(i-1)*(ip-1)+x]; }

    template<bool fwd, typename Tfd> Tfd *exec_(Tfd *cc, Tfd *ch, Tfd *buf,
      size_t nthreads) const
      {
      static const auto tifd = tidx<Tfd *>();
      if ((l1==1) && (ido==1))
        {
        Tfd *p1=cc, *p2=ch;
        if constexpr (fwd)
          for (auto it=passes.rbegin(); it!=passes.rend(); ++it)
            {
            auto res = static_cast<Tfd *>((*it)->exec(tifd,
              p1, p2, buf, fwd, nthreads));
            if (res==p2) swap(p1,p2);
            }
        else
          for (const auto &pass: passes)
            {
            auto res = static_cast<Tfd *>(pass->exec(tifd,
              p1, p2, buf, fwd, nthreads));
            if (res==p2) swap(p1,p2);
            }
        return p1;
        }
      else
        MR_fail("not yet supported");
      }

  public:
    rfft_multipass(size_t l1_, size_t ido_, size_t ip_,
      const Troots<Tfs> &roots, bool /*vectorize*/=false)
      : l1(l1_), ido(ido_), ip(ip_), bufsz(0), need_cpy(false),
        wa((ip-1)*(ido-1))
      {
      size_t N=ip*l1*ido;
      auto rfct = roots->size()/N;
      MR_assert(roots->size()==N*rfct, "mismatch");
      for (size_t j=1; j<ip; ++j)
        for (size_t i=1; i<=(ido-1)/2; ++i)
          {
          auto val = (*roots)[rfct*j*l1*i];
          wa[(j-1)*(ido-1)+2*i-2] = val.r;
          wa[(j-1)*(ido-1)+2*i-1] = val.i;
          }

      auto factors = rfftpass<Tfs>::factorize(ip);

      size_t l1l=1;
      for (auto fct: factors)
        {
        passes.push_back(rfftpass<Tfs>::make_pass(l1l, ip/(fct*l1l), fct, roots));
        l1l*=fct;
        }
      for (const auto &pass: passes)
        {
        bufsz = max(bufsz, pass->bufsize());
        need_cpy |= pass->needs_copy();
        }
      if ((l1!=1)||(ido!=1))
        {
        need_cpy=true;
        bufsz += 2*ip;
        }
      }

    virtual size_t bufsize() const { return bufsz; }
    virtual bool needs_copy() const { return need_cpy; }

    POCKETFFT_EXEC_DISPATCH
  };

template <typename Tfs> class rfftp_complexify: public rfftpass<Tfs>
  {
  private:
    size_t N;
    Troots<Tfs> roots;
    size_t rfct;
    Tcpass<Tfs> pass;
    size_t l1, ido;
    static constexpr size_t ip=2;

    template<bool fwd, typename Tfd> Tfd *exec_ (Tfd * DUCC0_RESTRICT cc,
      Tfd * DUCC0_RESTRICT ch, Tfd * buf, size_t nthreads) const
      {
      using Tcd = Cmplx<Tfd>;
      auto ccc = reinterpret_cast<Tcd *>(cc);
      auto cch = reinterpret_cast<Tcd *>(ch);
      auto cbuf = reinterpret_cast<Tcd *>(buf);
      static const auto ticd = tidx<Tcd *>();
      if constexpr(fwd)
        {
        auto res = static_cast<Tcd *>(pass->exec(ticd,
          ccc, cch, cbuf, true, nthreads));
        auto rres = (res==ccc) ? ch : cc;
        rres[0] = res[0].r+res[0].i;
//FIXME: parallelize?
        for (size_t i=1, xi=N/2-1; i<=xi; ++i, --xi)
          {
          auto xe = res[i]+res[xi].conj();
          auto xo = Tcd(res[i].i+res[xi].i, res[xi].r-res[i].r)
                  * (*roots)[rfct*i].conj();
          rres[2*i-1] = Tfs(0.5)*(xe.r+xo.r);
          rres[2*i] = Tfs(0.5)*(xe.i+xo.i);
          rres[2*xi-1] = Tfs(0.5)*(xe.r-xo.r);
          rres[2*xi] = Tfs(0.5)*(xo.i-xe.i);
          }
        rres[N-1] = res[0].r-res[0].i;
        return rres;
        }
      else
        {
        cch[0] = Tcd(cc[0]+cc[N-1], cc[0]-cc[N-1]);
//FIXME: parallelize?
        for (size_t i=1, xi=N/2-1; i<=xi; ++i, --xi)
          {
          Tcd t1 (cc[2*i-1], cc[2*i]);
          Tcd t2 (cc[2*xi-1], -cc[2*xi]);
          auto xe = t1+t2;
          auto xo = (t1-t2)*(*roots)[rfct*i];
          cch[i] = (xe + Tcd(-xo.i, xo.r));
          cch[xi] = (xe.conj() + Tcd(xo.i, xo.r));
          }
        auto res = static_cast<Tcd *>(pass->exec(ticd,
          cch, ccc, cbuf, false, nthreads));
        return (res==ccc) ? cc : ch;
        }
      }

  public:
    rfftp_complexify(size_t N_, const Troots<Tfs> &roots_, bool vectorize=false)
      : N(N_), roots(roots_), pass(cfftpass<Tfs>::make_pass(N/2, vectorize))
      {
      rfct = roots->size()/N;
      MR_assert(roots->size()==N*rfct, "mismatch");
      MR_assert((N&1)==0, "N must be even");
      }

    virtual size_t bufsize() const { return 2*pass->bufsize(); }
    virtual bool needs_copy() const { return true; }

    POCKETFFT_EXEC_DISPATCH
  };
#undef POCKETFFT_EXEC_DISPATCH

template<typename Tfs> Trpass<Tfs> rfftpass<Tfs>::make_pass(size_t l1,
  size_t ido, size_t ip, const Troots<Tfs> &roots, bool vectorize)
  {
  MR_assert(ip>=1, "no zero-sized FFTs");
  if (ip==1) return make_shared<rfftp1<Tfs>>();
  if ((ip>1000) && ((ip&1)==0))  // use complex transform
    {
    bool doit = vectorize&&((ip&7)==0);  // vecpass might be beneficial
    doit |= ip>10000;  // complex multipass might be beneficial
    if (!doit)
      {
      auto factors = rfftpass<Tfs>::factorize(ip);
      for (auto factor: factors)
        // complex Bluestein or larger prime factor functions might be beneficial
        if (factor>5) { doit=true; break; }
      }
    if (doit)
      return make_shared<rfftp_complexify<Tfs>>(ip, roots, vectorize);
    }
  auto factors=rfftpass<Tfs>::factorize(ip);
  if (factors.size()==1)
    {
    switch(ip)
      {
      case 2:
        return make_shared<rfftp2<Tfs>>(l1, ido, roots);
      case 3:
        return make_shared<rfftp3<Tfs>>(l1, ido, roots);
      case 4:
        return make_shared<rfftp4<Tfs>>(l1, ido, roots);
      case 5:
        return make_shared<rfftp5<Tfs>>(l1, ido, roots);
      default:
        if (ip<135)
          return make_shared<rfftpg<Tfs>>(l1, ido, ip, roots);
        else
          return make_shared<rfftpblue<Tfs>>(l1, ido, ip, roots, vectorize);
      }
    }
  else // more than one factor, need a multipass
    return make_shared<rfft_multipass<Tfs>>(l1, ido, ip, roots, vectorize);
  }

}}

#endif
