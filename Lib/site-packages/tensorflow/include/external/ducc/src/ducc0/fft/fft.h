/*
This file is part of the ducc FFT library

Copyright (C) 2010-2023 Max-Planck-Society
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

#ifndef DUCC0_FFT_H
#define DUCC0_FFT_H

#include <cstddef>
#include <typeindex>
#include <memory>
#include <vector>
#include <complex>
#include "ducc0/infra/error_handling.h"
#include "ducc0/infra/aligned_array.h"
#include "ducc0/infra/mav.h"
#include "ducc0/math/cmplx.h"
#include "ducc0/math/unity_roots.h"

namespace ducc0 {

namespace detail_fft {

using namespace std;

template<typename T> using Troots = shared_ptr<const UnityRoots<T,Cmplx<T>>>;
template<typename T> inline auto tidx() { return type_index(typeid(T)); }

template<typename T> inline void PM(T &a, T &b, T c, T d)
  { a=c+d; b=c-d; }
template<typename T> inline void PMINPLACE(T &a, T &b)
  { T t = a; a+=b; b=t-b; }
template<typename T> inline void MPINPLACE(T &a, T &b)
  { T t = a; a-=b; b=t+b; }
template<bool fwd, typename T, typename T2> void special_mul (const Cmplx<T> &v1, const Cmplx<T2> &v2, Cmplx<T> &res)
  {
  res = fwd ? Cmplx<T>(v1.r*v2.r+v1.i*v2.i, v1.i*v2.r-v1.r*v2.i)
            : Cmplx<T>(v1.r*v2.r-v1.i*v2.i, v1.r*v2.i+v1.i*v2.r);
  }

struct util1d // hack to avoid duplicate symbols
  {
  /* returns the smallest composite of 2, 3, 5, 7 and 11 which is >= n */
  DUCC0_NOINLINE static size_t good_size_cmplx(size_t n)
    {
    if (n<=12) return n;

    size_t bestfac=2*n;
    for (size_t f11=1; f11<bestfac; f11*=11)
      for (size_t f117=f11; f117<bestfac; f117*=7)
        for (size_t f1175=f117; f1175<bestfac; f1175*=5)
          {
          size_t x=f1175;
          while (x<n) x*=2;
          for (;;)
            {
            if (x<n)
              x*=3;
            else if (x>n)
              {
              if (x<bestfac) bestfac=x;
              if (x&1) break;
              x>>=1;
              }
            else
              return n;
            }
          }
    return bestfac;
    }
  /* returns the smallest composite of 2, 3, 5, 7 and 11 which is >= n
     and a multiple of required_factor. */
  DUCC0_NOINLINE static size_t good_size_cmplx(size_t n,
    size_t required_factor)
    {
    MR_assert(required_factor>=1, "required_factor must not be 0");
    return good_size_cmplx((n+required_factor-1)/required_factor) * required_factor;
    }

  /* returns the smallest composite of 2, 3, 5 which is >= n */
  DUCC0_NOINLINE static size_t good_size_real(size_t n)
    {
    if (n<=6) return n;

    size_t bestfac=2*n;
    for (size_t f5=1; f5<bestfac; f5*=5)
      {
      size_t x = f5;
      while (x<n) x *= 2;
      for (;;)
        {
        if (x<n)
          x*=3;
        else if (x>n)
          {
          if (x<bestfac) bestfac=x;
          if (x&1) break;
          x>>=1;
          }
        else
          return n;
        }
      }
    return bestfac;
    }
  /* returns the smallest composite of 2, 3, 5 which is >= n
     and a multiple of required_factor. */
  DUCC0_NOINLINE static size_t good_size_real(size_t n,
    size_t required_factor)
    {
    MR_assert(required_factor>=1, "required_factor must not be 0");
    return good_size_real((n+required_factor-1)/required_factor) * required_factor;
    }

  DUCC0_NOINLINE static vector<size_t> prime_factors(size_t N)
    {
    MR_assert(N>0, "need a positive number");
    vector<size_t> factors;
    while ((N&1)==0)
      { N>>=1; factors.push_back(2); }
    for (size_t divisor=3; divisor*divisor<=N; divisor+=2)
    while ((N%divisor)==0)
      {
      factors.push_back(divisor);
      N/=divisor;
      }
    if (N>1) factors.push_back(N);
    return factors;
    }
  };

// T: "type", f/c: "float/complex", s/v: "scalar/vector"
template <typename Tfs> class cfftpass
  {
  public:
    virtual ~cfftpass(){}
    using Tcs = Cmplx<Tfs>;

    // number of Tcd values required as scratch space during "exec"
    // will be provided in "buf"
    virtual size_t bufsize() const = 0;
    virtual bool needs_copy() const = 0;
    virtual void *exec(const type_index &ti, void *in, void *copy, void *buf,
      bool fwd, size_t nthreads=1) const = 0;

    static vector<size_t> factorize(size_t N)
      {
      MR_assert(N>0, "need a positive number");
      vector<size_t> factors;
      factors.reserve(15);
      while ((N&7)==0)
        { factors.push_back(8); N>>=3; }
      while ((N&3)==0)
        { factors.push_back(4); N>>=2; }
      if ((N&1)==0)
        {
        N>>=1;
        // factor 2 should be at the front of the factor list
        factors.push_back(2);
        swap(factors[0], factors.back());
        }
      for (size_t divisor=3; divisor*divisor<=N; divisor+=2)
      while ((N%divisor)==0)
        {
        factors.push_back(divisor);
        N/=divisor;
        }
      if (N>1) factors.push_back(N);
      return factors;
      }

    static shared_ptr<cfftpass> make_pass(size_t l1, size_t ido, size_t ip,
      const Troots<Tfs> &roots, bool vectorize=false);
    static shared_ptr<cfftpass> make_pass(size_t ip, bool vectorize=false)
      {
      return make_pass(1,1,ip,make_shared<UnityRoots<Tfs,Cmplx<Tfs>>>(ip),
        vectorize);
      }
  };

template <typename Tfs> class rfftpass
  {
  public:
    virtual ~rfftpass(){}

    // number of Tfd values required as scratch space during "exec"
    // will be provided in "buf"
    virtual size_t bufsize() const = 0;
    virtual bool needs_copy() const = 0;
    virtual void *exec(const type_index &ti, void *in, void *copy, void *buf,
      bool fwd, size_t nthreads=1) const = 0;

    static vector<size_t> factorize(size_t N)
      {
      MR_assert(N>0, "need a positive number");
      vector<size_t> factors;
      while ((N&3)==0)
        { factors.push_back(4); N>>=2; }
      if ((N&1)==0)
        {
        N>>=1;
        // factor 2 should be at the front of the factor list
        factors.push_back(2);
        swap(factors[0], factors.back());
        }
      for (size_t divisor=3; divisor*divisor<=N; divisor+=2)
      while ((N%divisor)==0)
        {
        factors.push_back(divisor);
        N/=divisor;
        }
      if (N>1) factors.push_back(N);
      return factors;
      }

    static shared_ptr<rfftpass> make_pass(size_t l1, size_t ido, size_t ip,
       const Troots<Tfs> &roots, bool vectorize=false);
    static shared_ptr<rfftpass> make_pass(size_t ip, bool vectorize=false)
      {
      return make_pass(1,1,ip,make_shared<UnityRoots<Tfs,Cmplx<Tfs>>>(ip),
        vectorize);
      }
  };

template<typename T> using Tcpass = shared_ptr<cfftpass<T>>;
template<typename T> using Trpass = shared_ptr<rfftpass<T>>;

template<typename Tfs> class pocketfft_c
  {
  private:
    size_t N;
    size_t critbuf;
    Tcpass<Tfs> plan;

  public:
    pocketfft_c(size_t n, bool vectorize=false)
      : N(n), critbuf(((N&1023)==0) ? 16 : 0),
        plan(cfftpass<Tfs>::make_pass(n,vectorize)) {}
    size_t length() const { return N; }
    size_t bufsize() const { return N*plan->needs_copy()+2*critbuf+plan->bufsize(); }
    template<typename Tfd> DUCC0_NOINLINE Cmplx<Tfd> *exec(Cmplx<Tfd> *in, Cmplx<Tfd> *buf,
      Tfs fct, bool fwd, size_t nthreads=1) const
      {
      static const auto tic = tidx<Cmplx<Tfd> *>();
      auto res = static_cast<Cmplx<Tfd> *>(plan->exec(tic,
        in, buf+critbuf+plan->bufsize(), buf+critbuf, fwd, nthreads));
      if (fct!=Tfs(1))
        for (size_t i=0; i<N; ++i) res[i]*=fct;
      return res;
      }
    template<typename Tfd> DUCC0_NOINLINE void exec_copyback(Cmplx<Tfd> *in, Cmplx<Tfd> *buf,
      Tfs fct, bool fwd, size_t nthreads=1) const
      {
      static const auto tic = tidx<Cmplx<Tfd> *>();
      auto res = static_cast<Cmplx<Tfd> *>(plan->exec(tic,
        in, buf, buf+N*plan->needs_copy(), fwd, nthreads));
      if (res==in)
        {
        if (fct!=Tfs(1))
          for (size_t i=0; i<N; ++i) in[i]*=fct;
        }
      else
        {
        if (fct!=Tfs(1))
          for (size_t i=0; i<N; ++i) in[i]=res[i]*fct;
        else
          copy_n(res, N, in);
        }
      }
    template<typename Tfd> DUCC0_NOINLINE void exec(Cmplx<Tfd> *in, Tfs fct, bool fwd, size_t nthreads=1) const
      {
      aligned_array<Cmplx<Tfd>> buf(N*plan->needs_copy()+plan->bufsize());
      exec_copyback(in, buf.data(), fct, fwd, nthreads);
      }
  };

template<typename Tfs> class pocketfft_r
  {
  private:
    size_t N;
    Trpass<Tfs> plan;

  public:
    pocketfft_r(size_t n, bool vectorize=false)
      : N(n), plan(rfftpass<Tfs>::make_pass(n,vectorize)) {}
    size_t length() const { return N; }
    size_t bufsize() const { return N*plan->needs_copy()+plan->bufsize(); }
    template<typename Tfd> DUCC0_NOINLINE Tfd *exec(Tfd *in, Tfd *buf, Tfs fct,
      bool fwd, size_t nthreads=1) const
      {
      static const auto tifd = tidx<Tfd *>();
      auto res = static_cast<Tfd *>(plan->exec(tifd, in, buf,
        buf+N*plan->needs_copy(), fwd, nthreads));
      if (fct!=Tfs(1))
        for (size_t i=0; i<N; ++i) res[i]*=fct;
      return res;
      }
    template<typename Tfd> DUCC0_NOINLINE void exec_copyback(Tfd *in, Tfd *buf,
      Tfs fct, bool fwd, size_t nthreads=1) const
      {
      static const auto tifd = tidx<Tfd *>();
      auto res = static_cast<Tfd *>(plan->exec(tifd, in, buf,
        buf+N*plan->needs_copy(), fwd, nthreads));
      if (res==in)
        {
        if (fct!=Tfs(1))
          for (size_t i=0; i<N; ++i) in[i]*=fct;
        }
      else
        {
        if (fct!=Tfs(1))
          for (size_t i=0; i<N; ++i) in[i]=res[i]*fct;
        else
          copy_n(res, N, in);
        }
      }
    template<typename Tfd> DUCC0_NOINLINE void exec(Tfd *in, Tfs fct, bool fwd,
      size_t nthreads=1) const
      {
      aligned_array<Tfd> buf(N*plan->needs_copy()+plan->bufsize());
      exec_copyback(in, buf.data(), fct, fwd, nthreads);
      }
  };

template<typename Tfs> class pocketfft_hartley
  {
  private:
    size_t N;
    Trpass<Tfs> plan;

  public:
    pocketfft_hartley(size_t n, bool vectorize=false)
      : N(n), plan(rfftpass<Tfs>::make_pass(n,vectorize)) {}
    size_t length() const { return N; }
    size_t bufsize() const { return N+plan->bufsize(); }
    template<typename Tfd> DUCC0_NOINLINE Tfd *exec(Tfd *in, Tfd *buf, Tfs fct,
      size_t nthreads=1) const
      {
      static const auto tifd = tidx<Tfd *>();
      auto res = static_cast<Tfd *>(plan->exec(tifd,
        in, buf, buf+N, true, nthreads));
      auto res2 = (res==buf) ? in : buf;
      res2[0] = fct*res[0];
      size_t i=1, i1=1, i2=N-1;
      for (i=1; i<N-1; i+=2, ++i1, --i2)
        {
        res2[i1] = fct*(res[i]+res[i+1]);
        res2[i2] = fct*(res[i]-res[i+1]);
        }
      if (i<N)
        res2[i1] = fct*res[i];

      return res2;
      }
    template<typename Tfd> DUCC0_NOINLINE void exec_copyback(Tfd *in, Tfd *buf,
      Tfs fct, size_t nthreads=1) const
      {
      auto res = exec(in, buf, fct, nthreads);
      if (res!=in)
        copy_n(res, N, in);
      }
    template<typename Tfd> DUCC0_NOINLINE void exec(Tfd *in, Tfs fct,
      size_t nthreads=1) const
      {
      aligned_array<Tfd> buf(N+plan->bufsize());
      exec_copyback(in, buf.data(), fct, nthreads);
      }
  };

template<typename Tfs> class pocketfft_fht
  {
  private:
    size_t N;
    Trpass<Tfs> plan;

  public:
    pocketfft_fht(size_t n, bool vectorize=false)
      : N(n), plan(rfftpass<Tfs>::make_pass(n,vectorize)) {}
    size_t length() const { return N; }
    size_t bufsize() const { return N+plan->bufsize(); }
    template<typename Tfd> DUCC0_NOINLINE Tfd *exec(Tfd *in, Tfd *buf, Tfs fct,
      size_t nthreads=1) const
      {
      static const auto tifd = tidx<Tfd *>();
      auto res = static_cast<Tfd *>(plan->exec(tifd,
        in, buf, buf+N, true, nthreads));
      auto res2 = (res==buf) ? in : buf;
      res2[0] = fct*res[0];
      size_t i=1, i1=1, i2=N-1;
      for (i=1; i<N-1; i+=2, ++i1, --i2)
        {
        res2[i1] = fct*(res[i]-res[i+1]);
        res2[i2] = fct*(res[i]+res[i+1]);
        }
      if (i<N)
        res2[i1] = fct*res[i];

      return res2;
      }
    template<typename Tfd> DUCC0_NOINLINE void exec_copyback(Tfd *in, Tfd *buf,
      Tfs fct, size_t nthreads=1) const
      {
      auto res = exec(in, buf, fct, nthreads);
      if (res!=in)
        copy_n(res, N, in);
      }
    template<typename Tfd> DUCC0_NOINLINE void exec(Tfd *in, Tfs fct,
      size_t nthreads=1) const
      {
      aligned_array<Tfd> buf(N+plan->bufsize());
      exec_copyback(in, buf.data(), fct, nthreads);
      }
  };

// R2R transforms using FFTW's halfcomplex format
template<typename Tfs> class pocketfft_fftw
  {
  private:
    size_t N;
    Trpass<Tfs> plan;

  public:
    pocketfft_fftw(size_t n, bool vectorize=false)
      : N(n), plan(rfftpass<Tfs>::make_pass(n,vectorize)) {}
    size_t length() const { return N; }
    size_t bufsize() const { return N+plan->bufsize(); }
    template<typename Tfd> DUCC0_NOINLINE Tfd *exec(Tfd *in, Tfd *buf, Tfs fct,
      bool fwd, size_t nthreads=1) const
      {
      static const auto tifd = tidx<Tfd *>();
      auto res = in;
      auto res2 = buf;
      if (!fwd) // go to FFTPACK halfcomplex order
        {
        res2[0] = fct*res[0];
        size_t i=1, i1=1, i2=N-1;
        for (i=1; i<N-1; i+=2, ++i1, --i2)
          {
          res2[i] = fct*res[i1];
          res2[i+1] = fct*res[i2];
          }
        if (i<N)
          res2[i] = fct*res[i1];
        swap(res, res2);
        }
      res = static_cast<Tfd *>(plan->exec(tifd,
        res, res2, buf+N, fwd, nthreads));
      if (!fwd) return res;

      // go to FFTW halfcomplex order
      res2 = (res==buf) ? in : buf;
      res2[0] = fct*res[0];
      size_t i=1, i1=1, i2=N-1;
      for (i=1; i<N-1; i+=2, ++i1, --i2)
        {
        res2[i1] = fct*res[i];
        res2[i2] = fct*res[i+1];
        }
      if (i<N)
        res2[i1] = fct*res[i];

      return res2;
      }
    template<typename Tfd> DUCC0_NOINLINE void exec_copyback(Tfd *in, Tfd *buf,
      Tfs fct, bool fwd, size_t nthreads=1) const
      {
      auto res = exec(in, buf, fct, fwd, nthreads);
      if (res!=in)
        copy_n(res, N, in);
      }
    template<typename Tfd> DUCC0_NOINLINE void exec(Tfd *in, Tfs fct, bool fwd,
      size_t nthreads=1) const
      {
      aligned_array<Tfd> buf(N+plan->bufsize());
      exec_copyback(in, buf.data(), fct, fwd, nthreads);
      }
  };

//
// sine/cosine transforms
//

template<typename T0> class T_dct1
  {
  private:
    pocketfft_r<T0> fftplan;

  public:
    DUCC0_NOINLINE T_dct1(size_t length, bool /*vectorize*/=false)
      : fftplan(2*(length-1)) {}

    template<typename T> DUCC0_NOINLINE T *exec(T c[], T buf[], T0 fct, bool ortho,
      int /*type*/, bool /*cosine*/, size_t nthreads=1) const
      {
      constexpr T0 sqrt2=T0(1.414213562373095048801688724209698L);
      size_t N=fftplan.length(), n=N/2+1;
      if (ortho)
        { c[0]*=sqrt2; c[n-1]*=sqrt2; }
      auto tmp=&buf[0];
      tmp[0] = c[0];
      for (size_t i=1; i<n; ++i)
        tmp[i] = tmp[N-i] = c[i];
      auto res = fftplan.exec(tmp, &buf[N], fct, true, nthreads);
      c[0] = res[0];
      for (size_t i=1; i<n; ++i)
        c[i] = res[2*i-1];
      if (ortho)
        { c[0]*=sqrt2*T0(0.5); c[n-1]*=sqrt2*T0(0.5); }
      return c;
      }
    template<typename T> DUCC0_NOINLINE void exec_copyback(T c[], T buf[], T0 fct, bool ortho,
      int /*type*/, bool /*cosine*/, size_t nthreads=1) const
      {
      exec(c, buf, fct, ortho, 1, true, nthreads);
      }
    template<typename T> DUCC0_NOINLINE void exec(T c[], T0 fct, bool ortho,
      int /*type*/, bool /*cosine*/, size_t nthreads=1) const
      {
      aligned_array<T> buf(bufsize());
      exec_copyback(c, buf.data(), fct, ortho, 1, true, nthreads);
      }

    size_t length() const { return fftplan.length()/2+1; }
    size_t bufsize() const { return fftplan.length()+fftplan.bufsize(); }
  };

template<typename T0> class T_dst1
  {
  private:
    pocketfft_r<T0> fftplan;

  public:
    DUCC0_NOINLINE T_dst1(size_t length, bool /*vectorize*/=false)
      : fftplan(2*(length+1)) {}

    template<typename T> DUCC0_NOINLINE T *exec(T c[], T buf[], T0 fct,
      bool /*ortho*/, int /*type*/, bool /*cosine*/, size_t nthreads=1) const
      {
      size_t N=fftplan.length(), n=N/2-1;
      auto tmp = &buf[0];
      tmp[0] = tmp[n+1] = c[0]*0;
      for (size_t i=0; i<n; ++i)
        { tmp[i+1]=c[i]; tmp[N-1-i]=-c[i]; }
      auto res = fftplan.exec(tmp, buf+N, fct, true, nthreads);
      for (size_t i=0; i<n; ++i)
        c[i] = -res[2*i+2];
      return c;
      }
    template<typename T> DUCC0_NOINLINE void exec_copyback(T c[], T buf[], T0 fct,
      bool /*ortho*/, int /*type*/, bool /*cosine*/, size_t nthreads=1) const
      {
      exec(c, buf, fct, true, 1, false, nthreads);
      }
    template<typename T> DUCC0_NOINLINE void exec(T c[], T0 fct,
      bool /*ortho*/, int /*type*/, bool /*cosine*/, size_t nthreads) const
      {
      aligned_array<T> buf(bufsize());
      exec_copyback(c, buf.data(), fct, true, 1, false, nthreads);
      }

    size_t length() const { return fftplan.length()/2-1; }
    size_t bufsize() const { return fftplan.length()+fftplan.bufsize(); }
  };

template<typename T0> class T_dcst23
  {
  private:
    pocketfft_r<T0> fftplan;
    std::vector<T0> twiddle;

  public:
    DUCC0_NOINLINE T_dcst23(size_t length, bool /*vectorize*/=false)
      : fftplan(length), twiddle(length)
      {
      UnityRoots<T0,Cmplx<T0>> tw(4*length);
      for (size_t i=0; i<length; ++i)
        twiddle[i] = tw[i+1].r;
      }

    template<typename T> DUCC0_NOINLINE T *exec(T c[], T buf[], T0 fct, bool ortho,
      int type, bool cosine, size_t nthreads=1) const
      {
      constexpr T0 sqrt2=T0(1.414213562373095048801688724209698L);
      size_t N=length();
      size_t NS2 = (N+1)/2;
      if (type==2)
        {
        c[0] *= 2;
        if ((N&1)==0) c[N-1]*=2;
        if (cosine)
          for (size_t k=1; k<N-1; k+=2)
            MPINPLACE(c[k+1], c[k]);
        else
          for (size_t k=1; k<N-1; k+=2)
            PMINPLACE(c[k+1], c[k]);
        if ((!cosine) && ((N&1)==0))
          c[N-1] *= -1;
        auto res = fftplan.exec(c, buf, fct, false, nthreads);
        c[0] = res[0];
        for (size_t k=1, kc=N-1; k<NS2; ++k, --kc)
          {
          T t1 = twiddle[k-1]*res[kc]+twiddle[kc-1]*res[k];
          T t2 = twiddle[k-1]*res[k]-twiddle[kc-1]*res[kc];
          c[k] = T0(0.5)*(t1+t2); c[kc]=T0(0.5)*(t1-t2);
          }
        if ((N&1)==0)
          c[NS2] = res[NS2]*twiddle[NS2-1];
        if (!cosine)  // swap order completely
          for (size_t k=0, kc=N-1; k<kc; ++k, --kc)
            std::swap(c[k], c[kc]);
        if (ortho)
          cosine ? c[0]*=sqrt2*T0(0.5) : c[N-1]*=sqrt2*T0(0.5);
        }
      else
        {
        if (ortho)
          cosine ? c[0]*=sqrt2 : c[N-1]*=sqrt2;
        if (!cosine)  // swap order completely
          for (size_t k=0, kc=N-1; k<NS2; ++k, --kc)
            std::swap(c[k], c[kc]);
        for (size_t k=1, kc=N-1; k<NS2; ++k, --kc)
          {
          T t1=c[k]+c[kc], t2=c[k]-c[kc];
          c[k] = twiddle[k-1]*t2+twiddle[kc-1]*t1;
          c[kc]= twiddle[k-1]*t1-twiddle[kc-1]*t2;
          }
        if ((N&1)==0)
          c[NS2] *= 2*twiddle[NS2-1];
        auto res = fftplan.exec(c, buf, fct, true, nthreads);
        if (res != c) // FIXME: not yet optimal
          copy_n(res, N, c);
        if ((!cosine) && ((N&1)==0))
          c[N-1] *= -1;
        if (cosine)
          for (size_t k=1; k<N-1; k+=2)
            MPINPLACE(c[k], c[k+1]);
        else
          for (size_t k=1; k<N-1; k+=2)
            PMINPLACE(c[k+1], c[k]);
        }
      return c;
      }
    template<typename T> DUCC0_NOINLINE void exec_copyback(T c[], T buf[], T0 fct,
      bool ortho, int type, bool cosine, size_t nthreads=1) const
      {
      exec(c, buf, fct, ortho, type, cosine, nthreads);
      }
    template<typename T> DUCC0_NOINLINE void exec(T c[], T0 fct, bool ortho,
      int type, bool cosine, size_t nthreads=1) const
      {
      aligned_array<T> buf(bufsize());
      exec(c, &buf[0], fct, ortho, type, cosine, nthreads);
      }

    size_t length() const { return fftplan.length(); }
    size_t bufsize() const { return fftplan.bufsize(); }
  };

template<typename T0> class T_dcst4
  {
  private:
    size_t N;
    std::unique_ptr<pocketfft_c<T0>> fft;
    std::unique_ptr<pocketfft_r<T0>> rfft;
    aligned_array<Cmplx<T0>> C2;
    size_t bufsz;

  public:
    DUCC0_NOINLINE T_dcst4(size_t length, bool /*vectorize*/=false)
      : N(length),
        fft((N&1) ? nullptr : make_unique<pocketfft_c<T0>>(N/2)),
        rfft((N&1)? make_unique<pocketfft_r<T0>>(N) : nullptr),
        C2((N&1) ? 0 : N/2),
        bufsz((N&1) ? (N+rfft->bufsize()) : (N+2*fft->bufsize()))
      {
      if ((N&1)==0)
        {
        UnityRoots<T0,Cmplx<T0>> tw(16*N);
        for (size_t i=0; i<N/2; ++i)
          C2[i] = tw[8*i+1].conj();
        }
      }

    template<typename T> DUCC0_NOINLINE T *exec(T c[], T buf[], T0 fct,
      bool /*ortho*/, int /*type*/, bool cosine, size_t nthreads) const
      {
      size_t n2 = N/2;
      if (!cosine)
        for (size_t k=0, kc=N-1; k<n2; ++k, --kc)
          std::swap(c[k], c[kc]);
      if (N&1)
        {
        // The following code is derived from the FFTW3 function apply_re11()
        // and is released under the 3-clause BSD license with friendly
        // permission of Matteo Frigo and Steven G. Johnson.

        auto y = buf;
        {
        size_t i=0, m=n2;
        for (; m<N; ++i, m+=4)
          y[i] = c[m];
        for (; m<2*N; ++i, m+=4)
          y[i] = -c[2*N-m-1];
        for (; m<3*N; ++i, m+=4)
          y[i] = -c[m-2*N];
        for (; m<4*N; ++i, m+=4)
          y[i] = c[4*N-m-1];
        for (; i<N; ++i, m+=4)
          y[i] = c[m-4*N];
        }

        auto res = rfft->exec(y, y+N, fct, true, nthreads);
        {
        auto SGN = [](size_t i)
           {
           constexpr T0 sqrt2=T0(1.414213562373095048801688724209698L);
           return (i&2) ? -sqrt2 : sqrt2;
           };
        c[n2] = res[0]*SGN(n2+1);
        size_t i=0, i1=1, k=1;
        for (; k<n2; ++i, ++i1, k+=2)
          {
          c[i    ] = res[2*k-1]*SGN(i1)     + res[2*k  ]*SGN(i);
          c[N -i1] = res[2*k-1]*SGN(N -i)   - res[2*k  ]*SGN(N -i1);
          c[n2-i1] = res[2*k+1]*SGN(n2-i)   - res[2*k+2]*SGN(n2-i1);
          c[n2+i1] = res[2*k+1]*SGN(n2+i+2) + res[2*k+2]*SGN(n2+i1);
          }
        if (k == n2)
          {
          c[i   ] = res[2*k-1]*SGN(i+1) + res[2*k]*SGN(i);
          c[N-i1] = res[2*k-1]*SGN(i+2) + res[2*k]*SGN(i1);
          }
        }

        // FFTW-derived code ends here
        }
      else
        {
        // even length algorithm from
        // https://www.appletonaudio.com/blog/2013/derivation-of-fast-dct-4-algorithm-based-on-dft/
        auto y2 = reinterpret_cast<Cmplx<T> *>(buf);
        for(size_t i=0; i<n2; ++i)
          {
          y2[i].Set(c[2*i],c[N-1-2*i]);
          y2[i] *= C2[i];
          }

        auto res = fft->exec(y2, y2+N/2, fct, true, nthreads);
        for(size_t i=0, ic=n2-1; i<n2; ++i, --ic)
          {
          c[2*i  ] = T0( 2)*(res[i ].r*C2[i ].r-res[i ].i*C2[i ].i);
          c[2*i+1] = T0(-2)*(res[ic].i*C2[ic].r+res[ic].r*C2[ic].i);
          }
        }
      if (!cosine)
        for (size_t k=1; k<N; k+=2)
          c[k] = -c[k];
      return c;
      }
    template<typename T> DUCC0_NOINLINE void exec_copyback(T c[], T buf[], T0 fct,
      bool /*ortho*/, int /*type*/, bool cosine, size_t nthreads=1) const
      {
      exec(c, buf, fct, true, 4, cosine, nthreads);
      }
    template<typename T> DUCC0_NOINLINE void exec(T c[], T0 fct,
      bool /*ortho*/, int /*type*/, bool cosine, size_t nthreads=1) const
      {
      aligned_array<T> buf(bufsize());
      exec(c, &buf[0], fct, true, 4, cosine, nthreads);
      }

    size_t length() const { return N; }
    size_t bufsize() const { return bufsz; }
  };

using shape_t=fmav_info::shape_t;
using stride_t=fmav_info::stride_t;

constexpr bool FORWARD  = true,
               BACKWARD = false;

/// Complex-to-complex Fast Fourier Transform
/** This executes a Fast Fourier Transform on \a in and stores the result in
 *  \a out.
 *
 *  \a in and \a out must have identical shapes; they may point to the same
 *  memory; in this case their strides must also be identical.
 *
 *  \a axes specifies the axes over which the transform is carried out.
 *
 *  If \a forward is true, a minus sign will be used in the exponent.
 *
 *  No normalization factors will be applied by default; if multiplication by
 *  a constant is desired, it can be supplied in \a fct.
 *
 *  If the underlying array has more than one dimension, the computation will
 *  be distributed over \a nthreads threads.
 */
template<typename T> DUCC0_NOINLINE void c2c(const cfmav<std::complex<T>> &in,
  const vfmav<std::complex<T>> &out, const shape_t &axes, bool forward,
  T fct, size_t nthreads=1);

/// Fast Discrete Cosine Transform
/** This executes a DCT on \a in and stores the result in \a out.
 *
 *  \a in and \a out must have identical shapes; they may point to the same
 *  memory; in this case their strides must also be identical.
 *
 *  \a axes specifies the axes over which the transform is carried out.
 *
 *  If \a forward is true, a DCT is computed, otherwise an inverse DCT.
 *
 *  \a type specifies the desired type (1-4) of the transform.
 *
 *  No normalization factors will be applied by default; if multiplication by
 *  a constant is desired, it can be supplied in \a fct.
 *
 *  If \a ortho is true, the first and last array entries are corrected (if
 *  necessary) to allow an orthonormalized transform.
 *
 *  If the underlying array has more than one dimension, the computation will
 *  be distributed over \a nthreads threads.
 */
template<typename T> DUCC0_NOINLINE void dct(const cfmav<T> &in, const vfmav<T> &out,
  const shape_t &axes, int type, T fct, bool ortho, size_t nthreads=1);

/// Fast Discrete Sine Transform
/** This executes a DST on \a in and stores the result in \a out.
 *
 *  \a in and \a out must have identical shapes; they may point to the same
 *  memory; in this case their strides must also be identical.
 *
 *  \a axes specifies the axes over which the transform is carried out.
 *
 *  If \a forward is true, a DST is computed, otherwise an inverse DST.
 *
 *  \a type specifies the desired type (1-4) of the transform.
 *
 *  No normalization factors will be applied by default; if multiplication by
 *  a constant is desired, it can be supplied in \a fct.
 *
 *  If \a ortho is true, the first and last array entries are corrected (if
 *  necessary) to allow an orthonormalized transform.
 *
 *  If the underlying array has more than one dimension, the computation will
 *  be distributed over \a nthreads threads.
 */
template<typename T> DUCC0_NOINLINE void dst(const cfmav<T> &in, const vfmav<T> &out,
  const shape_t &axes, int type, T fct, bool ortho, size_t nthreads=1);

template<typename T> DUCC0_NOINLINE void r2c(const cfmav<T> &in,
  const vfmav<std::complex<T>> &out, size_t axis, bool forward, T fct,
  size_t nthreads=1);

template<typename T> DUCC0_NOINLINE void r2c(const cfmav<T> &in,
  const vfmav<std::complex<T>> &out, const shape_t &axes,
  bool forward, T fct, size_t nthreads=1);

template<typename T> DUCC0_NOINLINE void c2r(const cfmav<std::complex<T>> &in,
  const vfmav<T> &out,  size_t axis, bool forward, T fct, size_t nthreads=1);

template<typename T> DUCC0_NOINLINE void c2r(const cfmav<std::complex<T>> &in,
  const vfmav<T> &out, const shape_t &axes, bool forward, T fct,
  size_t nthreads=1);

template<typename T> DUCC0_NOINLINE void c2r_mut(const vfmav<std::complex<T>> &in,
  const vfmav<T> &out, const shape_t &axes, bool forward, T fct,
  size_t nthreads=1);

template<typename T> DUCC0_NOINLINE void r2r_fftpack(const cfmav<T> &in,
  const vfmav<T> &out, const shape_t &axes, bool real2hermitian, bool forward,
  T fct, size_t nthreads=1);

template<typename T> DUCC0_NOINLINE void r2r_fftw(const cfmav<T> &in,
  const vfmav<T> &out, const shape_t &axes, bool forward,
  T fct, size_t nthreads=1);

template<typename T> DUCC0_NOINLINE void r2r_separable_hartley(const cfmav<T> &in,
  const vfmav<T> &out, const shape_t &axes, T fct, size_t nthreads=1);

template<typename T> DUCC0_NOINLINE void r2r_separable_fht(const cfmav<T> &in,
  const vfmav<T> &out, const shape_t &axes, T fct, size_t nthreads=1);

template<typename T> void r2r_genuine_hartley(const cfmav<T> &in,
  const vfmav<T> &out, const shape_t &axes, T fct, size_t nthreads=1);

template<typename T> void r2r_genuine_fht(const cfmav<T> &in,
  const vfmav<T> &out, const shape_t &axes, T fct, size_t nthreads=1);

/// Convolution and zero-padding/truncation along one axis
/** This performs a circular convolution with the kernel \a kernel on axis
 *  \a axis of \a in, applies the necessary zero-padding/truncation on this
 *  axis to give it the length \a out.shape(axis),and returns the result
 *  in \a out.
 *
 *  The main purpose of this routine is efficiency: the combination of the above
 *  operations can be carried out more quickly than running the individual
 *  operations in succession.
 *
 *  \a in and \a out must have identical shapes, with the possible exception
 *  of the axis \a axis; they may point to the same memory; in this case all
 *  of their strides must be identical.
 *
 *  \a axis specifies the axis over which the operation is carried out.
 *
 *  \a kernel must have the same length as \a in.shape(axis); it must be
 *  provided in the same domain as \a in (i.e. not pre-transformed).
 *
 *  If \a in has more than one dimension, the computation will
 *  be distributed over \a nthreads threads.
 */
template<typename T> DUCC0_NOINLINE void convolve_axis(const cfmav<T> &in,
  const vfmav<T> &out, size_t axis, const cmav<T,1> &kernel, size_t nthreads=1);

template<typename T> DUCC0_NOINLINE void convolve_axis(const cfmav<complex<T>> &in,
  const vfmav<complex<T>> &out, size_t axis, const cmav<complex<T>,1> &kernel,
  size_t nthreads=1);
}

using detail_fft::pocketfft_c;
using detail_fft::pocketfft_r;
using detail_fft::pocketfft_hartley;
using detail_fft::pocketfft_fht;
using detail_fft::pocketfft_fftw;

using detail_fft::FORWARD;
using detail_fft::BACKWARD;
using detail_fft::c2c;
using detail_fft::c2r;
using detail_fft::c2r_mut;
using detail_fft::r2c;
using detail_fft::r2r_fftpack;
using detail_fft::r2r_fftw;
using detail_fft::r2r_separable_hartley;
using detail_fft::r2r_genuine_hartley;
using detail_fft::r2r_separable_fht;
using detail_fft::r2r_genuine_fht;
using detail_fft::dct;
using detail_fft::dst;
using detail_fft::convolve_axis;

inline size_t good_size_complex(size_t n)
  { return detail_fft::util1d::good_size_cmplx(n); }
inline size_t good_size_real(size_t n)
  { return detail_fft::util1d::good_size_real(n); }
inline size_t good_size_complex(size_t n, size_t required_factor)
  { return detail_fft::util1d::good_size_cmplx(n, required_factor); }
inline size_t good_size_real(size_t n, size_t required_factor)
  { return detail_fft::util1d::good_size_real(n, required_factor); }

}

#endif
