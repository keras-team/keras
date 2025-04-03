/** \file ducc0/math/cmplx.h
 *  Minimalistic complex number class
 *
 *  \copyright Copyright (C) 2019-2023 Max-Planck-Society
 *  \author Martin Reinecke
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

#ifndef DUCC0_CMPLX_H
#define DUCC0_CMPLX_H

namespace ducc0 {

/// Very basic class representing complex numbers
/** Meant exclusively for internal low-level use, e.g. in FFT routines. */
template<typename T> struct Cmplx {
  T r, i;
  Cmplx() {}
  constexpr Cmplx(T r_, T i_) : r(r_), i(i_) {}
  constexpr Cmplx(T r_) : r(r_), i(T(0)) {}
  void Set(T r_, T i_) { r=r_; i=i_; }
  void Set(T r_) { r=r_; i=T(0); }
  void Split(T &r_, T &i_) const { r_=r; i_=i; }
  void SplitConj(T &r_, T &i_) const { r_=r; i_=-i; }
  Cmplx &operator+= (const Cmplx &other)
    { r+=other.r; i+=other.i; return *this; }
  template<typename T2>Cmplx &operator*= (T2 other)
    { r*=other; i*=other; return *this; }
  template<typename T2>Cmplx &operator*= (const Cmplx<T2> &other)
    {
    T tmp = r*other.r - i*other.i;
    i = r*other.i + i*other.r;
    r = tmp;
    return *this;
    }
  Cmplx conj() const { return {r, -i}; }
  template<typename T2>Cmplx &operator+= (const Cmplx<T2> &other)
    { r+=other.r; i+=other.i; return *this; }
  template<typename T2>Cmplx &operator-= (const Cmplx<T2> &other)
    { r-=other.r; i-=other.i; return *this; }
  template<typename T2> auto operator* (const T2 &other) const
    -> Cmplx<decltype(r*other)>
    { return {r*other, i*other}; }
  template<typename T2> auto operator+ (const Cmplx<T2> &other) const
    -> Cmplx<decltype(r+other.r)>
    { return {r+other.r, i+other.i}; }
  template<typename T2> auto operator- (const Cmplx<T2> &other) const
    -> Cmplx<decltype(r+other.r)>
    { return {r-other.r, i-other.i}; }
  template<typename T2> auto operator* (const Cmplx<T2> &other) const
    -> Cmplx<decltype(r+other.r)>
    { return {r*other.r-i*other.i, r*other.i + i*other.r}; }
  template<bool fwd, typename T2> auto special_mul (const Cmplx<T2> &other) const
    -> Cmplx<decltype(r+other.r)>
    {
    using Tres = Cmplx<decltype(r+other.r)>;
    return fwd ? Tres(r*other.r+i*other.i, i*other.r-r*other.i)
               : Tres(r*other.r-i*other.i, r*other.i+i*other.r);
    }
  };

}

#endif
