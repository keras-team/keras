/*! \file ducc0/infra/mav.h
 *  Classes for dealing with multidimensional arrays
 *
 *  \copyright Copyright (C) 2019-2024 Max-Planck-Society
 *  \author Martin Reinecke
 *  */

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

#ifndef DUCC0_MAV_H
#define DUCC0_MAV_H

#include <array>
#include <vector>
#include <memory>
#include <numeric>
#include <cstddef>
#include <functional>
#include <tuple>
#include "ducc0/infra/error_handling.h"
#include "ducc0/infra/aligned_array.h"
#include "ducc0/infra/misc_utils.h"
#include "ducc0/infra/threading.h"

namespace ducc0 {

namespace detail_mav {

using namespace std;

// the next line is necessary to address some sloppy name choices in AdaptiveCpp
using std::min, std::max;

struct uninitialized_dummy {};
constexpr uninitialized_dummy UNINITIALIZED;

template<typename T> class cmembuf
  {
  protected:
    shared_ptr<vector<T>> ptr;
    shared_ptr<quick_array<T>> rawptr;
    const T *d;

    cmembuf(const T *d_, const cmembuf &other)
      : ptr(other.ptr), rawptr(other.rawptr), d(d_) {}

    // externally owned data pointer
    cmembuf(const T *d_)
      : d(d_) {}
    // share another memory buffer, but read-only
    cmembuf(const cmembuf &other)
      : ptr(other.ptr), rawptr(other.rawptr), d(other.d) {}
    cmembuf(size_t sz)
      : ptr(make_shared<vector<T>>(sz)), d(ptr->data()) {}
#if 1
    cmembuf(size_t sz, uninitialized_dummy)
      : rawptr(make_shared<quick_array<T>>(sz)), d(rawptr->data()) {}
# else // "poison" the array with a fixed value; use for debugging
    cmembuf(size_t sz, uninitialized_dummy)
      : rawptr(make_shared<quick_array<T>>(sz)), d(rawptr->data())
      { for (size_t i=0; i<sz; ++i) (*rawptr)[i]=T(42000000); }
#endif
    // take over another memory buffer
    cmembuf(cmembuf &&other) = default;

  public:
    cmembuf(): d(nullptr) {}
    void assign(const cmembuf &other)
      {
      ptr = other.ptr;
      rawptr = other.rawptr;
      d = other.d;
      }
    // read access to element #i
    template<typename I> const T &raw(I i) const
      { return d[i]; }
    // read access to data area
    const T *data() const
      { return d; }
  };

constexpr size_t MAXIDX=~(size_t(0));

struct slice
  {
  size_t beg, end;
  ptrdiff_t step;
  slice() : beg(0), end(MAXIDX), step(1) {}
  slice(size_t idx) : beg(idx), end(idx), step(1) {}
  slice(size_t beg_, size_t end_, ptrdiff_t step_=1)
    : beg(beg_), end(end_), step(step_)
    {
// FIXME: add sanity checks here
    }

  size_t size(size_t shp) const
    {
    if (beg==end) return 0;
    if (step>0) return (min(shp,end)-beg+step-1)/step;
    // negative step
    if (end==MAXIDX)
      return (beg-step)/(-step);
    return (beg-end-step-1)/(-step);
    }
  };

/// Helper class containing shape and stride information of an `fmav` object
class fmav_info
  {
  public:
    /// vector of nonnegative integers for storing the array shape
    using shape_t = vector<size_t>;
    /// vector of integers for storing the array strides
    using stride_t = vector<ptrdiff_t>;

  protected:
    shape_t shp;
    stride_t str;
    size_t sz;

    static stride_t shape2stride(const shape_t &shp)
      {
      auto ndim = shp.size();
      // MR using the static_cast just to avoid a GCC warning.
//      stride_t res(ndim);
      stride_t res(static_cast<int>(ndim));
      if (ndim==0) return res;
      res[ndim-1]=1;
      for (size_t i=2; i<=ndim; ++i)
        res[ndim-i] = res[ndim-i+1]*ptrdiff_t(shp[ndim-i+1]);
      return res;
      }
    template<typename... Ns> ptrdiff_t getIdx(size_t dim, size_t n, Ns... ns) const
      { return str[dim]*ptrdiff_t(n) + getIdx(dim+1, ns...); }
    ptrdiff_t getIdx(size_t dim, size_t n) const
      { return str[dim]*ptrdiff_t(n); }
    ptrdiff_t getIdx(size_t /*dim*/) const
      { return 0; }

  public:
    /// Constructs a 1D object with all extents and strides set to zero.
    fmav_info() : shp(1,0), str(1,0), sz(0) {}
    /// Constructs an object with the given shape and stride.
    fmav_info(const shape_t &shape_, const stride_t &stride_)
      : shp(shape_), str(stride_),
        sz(accumulate(shp.begin(),shp.end(),size_t(1),multiplies<>()))
      {
      MR_assert(shp.size()==str.size(), "dimensions mismatch");
      }
    /// Constructs an object with the given shape and computes the strides
    /// automatically, assuming a C-contiguous memory layout.
    fmav_info(const shape_t &shape_)
      : fmav_info(shape_, shape2stride(shape_)) {}
    void assign(const fmav_info &other)
      {
      shp = other.shp;
      str = other.str;
      sz = other.sz;
      }
    /// Returns the dimensionality of the object.
    size_t ndim() const { return shp.size(); }
    /// Returns the total number of entries in the object.
    size_t size() const { return sz; }
    /// Returns the shape of the object.
    const shape_t &shape() const { return shp; }
    /// Returns the length along dimension \a i.
    size_t shape(size_t i) const { return shp[i]; }
    /// Returns the strides of the object.
    const stride_t &stride() const { return str; }
    /// Returns the stride along dimension \a i.
    const ptrdiff_t &stride(size_t i) const { return str[i]; }
    /// Returns true iff the last dimension has stride 1.
    /**  Typically used for optimization purposes. */
    bool last_contiguous() const
      { return ((ndim()==0) || (str.back()==1)); }
    /** Returns true iff the object is C-contiguous, i.e. if the stride of the
     *  last dimension is 1, the stride for the next-to-last dimension is the
     *  shape of the last dimension etc. */
    bool contiguous() const
      {
      auto ndim = shp.size();
      ptrdiff_t stride=1;
      for (size_t i=0; i<ndim; ++i)
        {
        if ((shp[ndim-1-i]!=1) && (str[ndim-1-i]!=stride))
          return false;
        stride *= ptrdiff_t(shp[ndim-1-i]);
        }
      return true;
      }
    /// Returns true iff this->shape and \a other.shape match.
    bool conformable(const fmav_info &other) const
      { return shp==other.shp; }
    /// Returns the one-dimensional index of an entry from the given
    /// multi-dimensional index tuple, taking strides into account.
    template<typename... Ns> ptrdiff_t idx(Ns... ns) const
      {
      MR_assert(ndim()==sizeof...(ns), "incorrect number of indices");
      return getIdx(0, ns...);
      }
    ptrdiff_t idx(const shape_t &ns) const
      {
      MR_assert(ndim()==ns.size(), "incorrect number of indices");
      size_t res = 0;
      for (size_t i=0; i<ndim(); ++i) res += str[i]*ns[i];
      return res;
      }
    template<typename RAiter> ptrdiff_t idxval(RAiter beg, RAiter end) const
      {
      MR_assert(ndim()==size_t(end-beg), "incorrect number of indices");
      size_t res = 0;
      for (size_t i=0; i<ndim(); ++i, ++beg) res += str[i]* (*beg);
      return res;
      }
    /// Returns the common broadcast shape of *this and \a shp2
    shape_t bcast_shape(const shape_t &shp2) const
      {
      shape_t res(max(shp.size(), shp2.size()), 1);
      for (size_t i=0; i<shp.size(); ++i)
        res[i+res.size()-shp.size()] = shp[i];
      for (size_t i=0; i<shp2.size(); ++i)
        {
        size_t i2 = i+res.size()-shp2.size();
        if (res[i2]==1)
          res[i2] = shp2[i];
        else
          MR_assert((res[i2]==shp2[i])||(shp2[i]==1),
            "arrays cannot be broadcast together");
        }
      return res;
      }
    void bcast_to_shape(const shape_t &shp2)
      {
      MR_assert(shp2.size()>=shp.size(), "cannot reduce dimensionality");
      stride_t newstr(shp2.size(), 0);
      for (size_t i=0; i<shp.size(); ++i)
        {
        size_t i2 = i+shp2.size()-shp.size();
        if (shp[i]!=1)
          {
          MR_assert(shp[i]==shp2[i2], "arrays cannot be broadcast together");
          newstr[i2] = str[i];
          }
        }
      shp = shp2;
      str = newstr;
      }

    void swap_axes(size_t ax0, size_t ax1)
      {
      MR_assert(ax0<=ndim() && ax1<=ndim(), "bad axes");
      if (ax0==ax1) return;
      swap(shp[ax0], shp[ax1]);
      swap(str[ax0], str[ax1]);
      }

    fmav_info extend_and_broadcast(const shape_t &new_shape,
      const shape_t &axpos) const
      {
      MR_assert(new_shape.size()>=ndim(),
        "new shape smaller than original one");
      MR_assert(axpos.size()==ndim(), "bad axpos size");
      stride_t new_stride(new_shape.size(), 0);
      vector<uint8_t> used(new_shape.size(),0);
      for (size_t i=0; i<ndim(); ++i)
        {
        MR_assert(axpos[i]<new_shape.size(), "bad axis number");
        MR_assert(shp[i]==new_shape[axpos[i]], "axis length nismatch");
        MR_assert(used[axpos[i]]==0, "repeated axis position");
        used[axpos[i]]=1;
        new_stride[axpos[i]] = str[i];
        }
      return fmav_info(new_shape, new_stride);
      }
    fmav_info extend_and_broadcast(const shape_t &new_shape,
      size_t firstaxis) const
      {
      shape_t axpos(ndim());
      std::iota(axpos.begin(), axpos.end(), firstaxis);
      return extend_and_broadcast(new_shape, axpos);
      }
    fmav_info transpose() const
      {
      return fmav_info({shp.crend(), shp.crbegin()}, {str.crbegin(), str.crend()});
      }
  protected:
    auto subdata(const vector<slice> &slices) const
      {
      auto ndim = shp.size();
      shape_t nshp(ndim);
      stride_t nstr(ndim);
      MR_assert(slices.size()==ndim, "incorrect number of slices");
      size_t n0=0;
      for (auto x:slices) if (x.beg==x.end) ++n0;
      ptrdiff_t nofs=0;
      nshp.resize(ndim-n0);
      nstr.resize(ndim-n0);
      for (size_t i=0, i2=0; i<ndim; ++i)
        {
        MR_assert(slices[i].beg<shp[i], "bad subset");
        nofs+=slices[i].beg*str[i];
        if (slices[i].beg!=slices[i].end)
          {
          auto ext = slices[i].size(shp[i]);
          MR_assert(slices[i].beg+(ext-1)*slices[i].step<shp[i], "bad subset");
          nshp[i2]=ext; nstr[i2]=slices[i].step*str[i];
          ++i2;
          }
        }
      return make_tuple(fmav_info(nshp, nstr), nofs);
      }
  };

/// Helper class containing shape and stride information of a `mav` object
template<size_t ndim> class mav_info
  {
  public:
    /// Fixed-size array of nonnegative integers for storing the array shape
    using shape_t = array<size_t, ndim>;
    /// Fixed-size array of integers for storing the array strides
    using stride_t = array<ptrdiff_t, ndim>;

  protected:
    shape_t shp;
    stride_t str;
    size_t sz;

    static stride_t shape2stride(const shape_t &shp)
      {
      stride_t res;
      if (ndim==0) return res;
      res[ndim-1]=1;
      for (size_t i=2; i<=ndim; ++i)
        res[ndim-i] = res[ndim-i+1]*ptrdiff_t(shp[ndim-i+1]);
      return res;
      }
    template<typename... Ns> ptrdiff_t getIdx(size_t dim, size_t n, Ns... ns) const
      { return str[dim]*n + getIdx(dim+1, ns...); }
    ptrdiff_t getIdx(size_t dim, size_t n) const
      { return str[dim]*n; }
    ptrdiff_t getIdx(size_t /*dim*/) const
      { return 0; }

  public:
    /// Constructs an object with all extents and strides set to zero.
    mav_info() : sz(0)
      {
      for (size_t i=0; i<ndim; ++i)
        { shp[i]=0; str[i]=0; }
      }
    /// Constructs an object with the given shape and stride.
    mav_info(const shape_t &shape_, const stride_t &stride_)
      : shp(shape_), str(stride_),
        sz(accumulate(shp.begin(),shp.end(),size_t(1),multiplies<>())) {}
    /// Constructs an object with the given shape and computes the strides
    /// automatically, assuming a C-contiguous memory layout.
    mav_info(const shape_t &shape_)
      : mav_info(shape_, shape2stride(shape_)) {}
    mav_info(const fmav_info &inp)
      {
      MR_assert(inp.ndim()==ndim, "dimensionality mismatch");
      sz=1;
      for (size_t i=0; i<ndim; ++i)
        {
        shp[i] = inp.shape(i);
        sz *= shp[i];
        str[i] = inp.stride(i);
        }
      }
    void assign(const mav_info &other)
      {
      shp = other.shp;
      str = other.str;
      sz = other.sz;
      }
    /// Returns the total number of entries in the object.
    size_t size() const { return sz; }
    /// Returns the shape of the object.
    const shape_t &shape() const { return shp; }
    /// Returns the length along dimension \a i.
    size_t shape(size_t i) const { return shp[i]; }
    /// Returns the strides of the object.
    const stride_t &stride() const { return str; }
    /// Returns the stride along dimension \a i.
    const ptrdiff_t &stride(size_t i) const { return str[i]; }
    /// Returns true iff the last dimension has stride 1.
    /**  Typically used for optimization purposes. */
    bool last_contiguous() const
      { return ((ndim==0) || (str.back()==1)); }
    /** Returns true iff the object is C-contiguous, i.e. if the stride of the
     *  last dimension is 1, the stride for the next-to-last dimension is the
     *  shape of the last dimension etc. */
    bool contiguous() const
      {
      ptrdiff_t stride=1;
      for (size_t i=0; i<ndim; ++i)
        {
        if ((shp[ndim-1-i]!=1) && (str[ndim-1-i]!=stride))
          return false;
        stride *= ptrdiff_t(shp[ndim-1-i]);
        }
      return true;
      }
    /// Returns true iff this->shape and \a other.shape match.
    bool conformable(const mav_info &other) const
      { return shp==other.shp; }
    /// Returns true iff this->shape and \a other match.
    bool conformable(const shape_t &other) const
      { return shp==other; }
    /// Returns the one-dimensional index of an entry from the given
    /// multi-dimensional index tuple, taking strides into account.
    template<typename... Ns> ptrdiff_t idx(Ns... ns) const
      {
      static_assert(ndim==sizeof...(ns), "incorrect number of indices");
      return getIdx(0, ns...);
      }
    mav_info transpose() const
      {
      shape_t shp2;
      stride_t str2;
      for (size_t i=0; i<ndim; ++i)
        {
        shp2[i] = shp[ndim-1-i];
        str2[i] = str[ndim-1-i];
        }
      return mav_info(shp2, str2);
      }
    mav_info<ndim+1> prepend_1() const
      {
      typename mav_info<ndim+1>::shape_t newshp;
      typename mav_info<ndim+1>::stride_t newstr;
      newshp[0] = 1;
      newstr[0] = 0;
      for (size_t i=0; i<ndim; ++i)
        {
        newshp[i+1] = shp[i];
        newstr[i+1] = str[i];
        }
      return mav_info<ndim+1>(newshp, newstr);
      }

  protected:
    template<size_t nd2> auto subdata(const vector<slice> &slices) const
      {
      MR_assert(slices.size()==ndim, "bad number of slices");
      array<size_t, nd2> nshp;
      array<ptrdiff_t, nd2> nstr;

      // unnecessary, but gcc warns otherwise
      for (size_t i=0; i<nd2; ++i) nshp[i]=nstr[i]=0;

      size_t n0=0;
      for (auto x:slices) if (x.beg==x.end) ++n0;
      MR_assert(n0+nd2==ndim, "bad extent");
      ptrdiff_t nofs=0;
      for (size_t i=0, i2=0; i<ndim; ++i)
        {
        MR_assert(slices[i].beg<shp[i], "bad subset");
        nofs+=slices[i].beg*str[i];
        if (slices[i].beg!=slices[i].end)
          {
          auto ext = slices[i].size(shp[i]);
          MR_assert(slices[i].beg+(ext-1)*slices[i].step<shp[i], "bad subset");
          nshp[i2]=ext; nstr[i2]=slices[i].step*str[i];
          ++i2;
          }
        }
      return make_tuple(mav_info<nd2>(nshp, nstr), nofs);
      }
  };

template<typename T> class cfmav: public fmav_info, public cmembuf<T>
  {
  protected:
    using tbuf = cmembuf<T>;
    using tinfo = fmav_info;
    using fmav_info::idx;

  public:
    using typename tinfo::shape_t;
    using typename tinfo::stride_t;
    using tbuf::raw, tbuf::data;


  protected:
    cfmav(const shape_t &shp_)
      : tinfo(shp_), tbuf(size()) {}
    cfmav(const shape_t &shp_, uninitialized_dummy)
      : tinfo(shp_), tbuf(size(), UNINITIALIZED) {}
    cfmav(const shape_t &shp_, const stride_t &str_, uninitialized_dummy)
      : tinfo(shp_, str_), tbuf(size(), UNINITIALIZED)
      {
      ptrdiff_t ofs=0;
      for (size_t i=0; i<ndim(); ++i)
        ofs += (ptrdiff_t(shp[i])-1)*str[i];
      MR_assert(ofs+1==ptrdiff_t(size()), "array is not compact");
      }
    cfmav(const fmav_info &info, const tbuf &buf)
      : tinfo(info), tbuf(buf) {}
    cfmav(const fmav_info &info, const T *d_, const tbuf &buf)
      : tinfo(info), tbuf(d_, buf) {}

  public:
    cfmav() {}
    cfmav(const T *d_, const shape_t &shp_, const stride_t &str_)
      : tinfo(shp_, str_), tbuf(d_) {}
    cfmav(const T *d_, const shape_t &shp_)
      : tinfo(shp_), tbuf(d_) {}
    cfmav(const T* d_, const tinfo &info)
      : tinfo(info), tbuf(d_) {}

    cfmav(const tbuf &buf, const shape_t &shp_, const stride_t &str_)
      : tinfo(shp_, str_), tbuf(buf) {}

    // no-op. Needed for template tricks.
    cfmav to_fmav() const { return *this; }

    void assign(const cfmav &other)
      {
      tinfo::assign(other);
      tbuf::assign(other);
      }

    /// Returns the data entry at the given set of indices.
    template<typename... Ns> const T &operator()(Ns... ns) const
      { return raw(idx(ns...)); }
    const T &operator()(const shape_t &ns) const
      { return raw(idx(ns)); }
    template<typename RAiter> const T& val(RAiter beg, RAiter end) const
      { return raw(idxval(beg, end)); }

    cfmav subarray(const vector<slice> &slices) const
      {
      auto [ninfo, nofs] = subdata(slices);
      return cfmav(ninfo, tbuf::d+nofs, *this);
      }
    cfmav extend_and_broadcast(const shape_t &new_shape, const shape_t &axpos) const
      {
      return cfmav(fmav_info::extend_and_broadcast(new_shape, axpos), *this);
      }
    cfmav extend_and_broadcast(const shape_t &new_shape, size_t firstaxis) const
      {
      return cfmav(fmav_info::extend_and_broadcast(new_shape, firstaxis), *this);
      }
    cfmav transpose() const
      {
      return cfmav(static_cast<const tinfo *>(this)->transpose(), *static_cast<const tbuf *>(this));
      }
  };

template<typename T> cfmav<T> subarray
  (const cfmav<T> &arr, const vector<slice> &slices)  
  { return arr.subarray(slices); }

template<typename T> class vfmav: public cfmav<T>
  {
  protected:
    using tbuf = cmembuf<T>;
    using tinfo = fmav_info;
    using tinfo::shp, tinfo::str;
    using fmav_info::idx;

  public:
    using typename tinfo::shape_t;
    using typename tinfo::stride_t;
    using tinfo::size, tinfo::shape, tinfo::stride;

  protected:
    vfmav(const fmav_info &info, const tbuf &buf)
      : cfmav<T>(info, buf) {}
    vfmav(const fmav_info &info, T *d_, const tbuf &buf)
      : cfmav<T>(info, d_, buf) {}

  public:
    using tbuf::raw, tbuf::data, tinfo::ndim;
    vfmav() {}
    vfmav(T *d_, const fmav_info &info)
      : cfmav<T>(d_, info) {}
    vfmav(T *d_, const shape_t &shp_, const stride_t &str_)
      : cfmav<T>(d_, shp_, str_) {}
    vfmav(T *d_, const shape_t &shp_)
      : cfmav<T>(d_, shp_) {}
    vfmav(const shape_t &shp_)
      : cfmav<T>(shp_) {}
    vfmav(const shape_t &shp_, uninitialized_dummy)
      : cfmav<T>(shp_, UNINITIALIZED) {}
    vfmav(const shape_t &shp_, const stride_t &str_, uninitialized_dummy)
      : cfmav<T>(shp_, str_, UNINITIALIZED)
      {
      ptrdiff_t ofs=0;
      for (size_t i=0; i<ndim(); ++i)
        ofs += (ptrdiff_t(shp[i])-1)*str[i];
      MR_assert(ofs+1==ptrdiff_t(size()), "array is not compact");
      }
    vfmav(tbuf &buf, const shape_t &shp_, const stride_t &str_)
      : cfmav<T>(buf, shp_, str_) {}

    T *data() const
     { return const_cast<T *>(tbuf::d); }
    template<typename I> T &raw(I i) const
      { return data()[i]; }

    // no-op. Needed for template tricks.
    using cfmav<T>::to_fmav;
    vfmav to_fmav() const { return *this; }

    void assign(const vfmav &other)
      {
      fmav_info::assign(other);
      cmembuf<T>::assign(other);
      }

    using cfmav<T>::operator();
    template<typename... Ns> T &operator()(Ns... ns) const
      { return raw(idx(ns...)); }
    T &operator()(const shape_t &ns) const
      { return raw(idx(ns)); }
    using cfmav<T>::val;
    template<typename RAiter> T& val(RAiter beg, RAiter end) const
      { return raw(idxval(beg, end)); }

    vfmav subarray(const vector<slice> &slices) const
      {
      auto [ninfo, nofs] = tinfo::subdata(slices);
      return vfmav(ninfo, data()+nofs, *this);
      }
    /** Returns a writable fmav with the specified shape.
     *  The strides are chosen in such a way that critical strides (multiples
     *  of 4096 bytes) along any dimension are avoided, by enlarging the
     *  allocated memory slightly if necessary.
     *  The array data is default-initialized. */
    static vfmav build_noncritical(const shape_t &shape)
      {
      auto ndim = shape.size();
      auto shape2 = noncritical_shape(shape, sizeof(T));
      vfmav tmp(shape2);
      vector<slice> slc(ndim);
      for (size_t i=0; i<ndim; ++i) slc[i] = slice(0, shape[i]);
      return tmp.subarray(slc);
      }
    /** Returns a writable fmav with the specified shape.
     *  The strides are chosen in such a way that critical strides (multiples
     *  of 4096 bytes) along any dimension are avoided, by enlarging the
     *  allocated memory slightly if necessary.
     *  The array data is not initialized. */
    static vfmav build_noncritical(const shape_t &shape, uninitialized_dummy)
      {
      auto ndim = shape.size();
      if (ndim<=1) return vfmav(shape, UNINITIALIZED);
      auto shape2 = noncritical_shape(shape, sizeof(T));
      vfmav tmp(shape2, UNINITIALIZED);
      vector<slice> slc(ndim);
      for (size_t i=0; i<ndim; ++i) slc[i] = slice(0, shape[i]);
      return tmp.subarray(slc);
      }
    vfmav extend_and_broadcast(const shape_t &new_shape, const shape_t &axpos) const
      {
      return vfmav(fmav_info::extend_and_broadcast(new_shape, axpos), *this);
      }
    vfmav extend_and_broadcast(const shape_t &new_shape, size_t firstaxis) const
      {
      return vfmav(fmav_info::extend_and_broadcast(new_shape, firstaxis), *this);
      }
    vfmav transpose() const
      {
      return vfmav(static_cast<const tinfo *>(this)->transpose(), *static_cast<const tbuf *>(this));
      }
  };

template<typename T> vfmav<T> subarray
  (const vfmav<T> &arr, const vector<slice> &slices)  
  { return arr.subarray(slices); }

template<typename T, size_t ndim> class cmav: public mav_info<ndim>, public cmembuf<T>
  {
  protected:
    template<typename T2, size_t nd2> friend class cmav;
    template<typename T2, size_t nd2> friend class vmav;

    using tinfo = mav_info<ndim>;
    using tbuf = cmembuf<T>;
    using tinfo::shp, tinfo::str;

  public:
    using typename tinfo::shape_t;
    using typename tinfo::stride_t;
    using tbuf::raw, tbuf::data;
    using tinfo::contiguous, tinfo::size, tinfo::idx, tinfo::conformable;

  protected:
    cmav() {}
    cmav(const shape_t &shp_, uninitialized_dummy)
      : tinfo(shp_), tbuf(size(), UNINITIALIZED) {}
    cmav(const shape_t &shp_)
      : tinfo(shp_), tbuf(size()) {}
    cmav(const tbuf &buf, const shape_t &shp_, const stride_t &str_)
      : tinfo(shp_, str_), tbuf(buf) {}
    cmav(const tinfo &info, const T *d_, const tbuf &buf)
      : tinfo(info), tbuf(d_, buf) {}
    cmav(const tinfo &info, const tbuf &buf)
      : tinfo(info), tbuf(buf) {}

  public:
    cmav(const T *d_, const shape_t &shp_, const stride_t &str_)
      : tinfo(shp_, str_), tbuf(d_) {}
    cmav(const T *d_, const shape_t &shp_)
      : tinfo(shp_), tbuf(d_) {}
    cmav(const cfmav<T> &inp)
      : tinfo(inp), tbuf(inp) {}
    void assign(const cmav &other)
      {
      mav_info<ndim>::assign(other);
      cmembuf<T>::assign(other);
      }
    operator cfmav<T>() const
      {
      return cfmav<T>(*this, {shp.begin(), shp.end()}, {str.begin(), str.end()});
      }
    // Needed for template tricks.
    cfmav<T> to_fmav() const { return operator cfmav<T>(); }

    template<typename... Ns> const T &operator()(Ns... ns) const
      { return raw(idx(ns...)); }
    template<size_t nd2> cmav<T,nd2> subarray(const vector<slice> &slices) const
      {
      auto [ninfo, nofs] = tinfo::template subdata<nd2> (slices);
      return cmav<T,nd2> (ninfo, tbuf::d+nofs, *this);
      }

    static cmav build_uniform(const shape_t &shape, const T &value)
      {
      // Don't do this at home!
      shape_t tshp;
      tshp.fill(1);
      cmav tmp(tshp);
      const_cast<T &>(tmp.raw(0)) = value;
      stride_t nstr;
      nstr.fill(0);
      return cmav(tmp, shape, nstr);
      }
    cmav transpose() const
      {
      return cmav(static_cast<const tinfo *>(this)->transpose(), *static_cast<const tbuf *>(this));
      }
    cmav<T, ndim+1> prepend_1() const
      {
      return cmav<T, ndim+1>(static_cast<const tinfo *>(this)->prepend_1(), *static_cast<const tbuf *>(this));
      }
    template<size_t ndim2> cmav<T, ndim2> reinterpret
      (const typename cmav<T, ndim2>::shape_t &newshp,
       const typename cmav<T, ndim2>::stride_t &newstr) const
      {
      return cmav<T, ndim2>(*static_cast<const tbuf *>(this), newshp, newstr);
      }
  };
template<size_t nd2, typename T, size_t ndim> cmav<T,nd2> subarray
  (const cmav<T, ndim> &arr, const vector<slice> &slices)  
  { return arr.template subarray<nd2>(slices); }

template<typename T, size_t ndim> class vmav: public cmav<T, ndim>
  {
  protected:
    template<typename T2, size_t nd2> friend class vmav;

    using parent = cmav<T, ndim>;
    using tinfo = mav_info<ndim>;
    using tbuf = cmembuf<T>;
    using tinfo::shp, tinfo::str;

  public:
    using typename tinfo::shape_t;
    using typename tinfo::stride_t;
    using tbuf::raw, tbuf::data;
    using tinfo::contiguous, tinfo::size, tinfo::idx, tinfo::conformable;

  protected:
    vmav(const tinfo &info, T *d_, const tbuf &buf)
      : parent(info, d_, buf) {}
    vmav(const tinfo &info, const tbuf &buf)
      : parent(info, buf) {}
    vmav(const tbuf &buf, const shape_t &shp_, const stride_t &str_)
      : parent(buf, shp_, str_){}

  public:
    vmav() {}
    vmav(T *d_, const shape_t &shp_, const stride_t &str_)
      : parent(d_, shp_, str_) {}
    vmav(T *d_, const shape_t &shp_)
      : parent(d_, shp_) {}
    vmav(const shape_t &shp_)
      : parent(shp_) {}
    vmav(const shape_t &shp_, uninitialized_dummy)
      : parent(shp_, UNINITIALIZED) {}
    vmav(const vfmav<T> &inp)
      : parent(inp) {}
      
    void assign(vmav &other)
      { parent::assign(other); }
    void dealloc()
      {
      vmav empty;
      assign(empty);
      }
    operator vfmav<T>() const
      {
      return vfmav<T>(*const_cast<tbuf *>(static_cast<const tbuf *>(this)), {shp.begin(), shp.end()}, {str.begin(), str.end()});
      }
    // Needed for template tricks.
    using cmav<T, ndim>::to_fmav;
    vfmav<T> to_fmav() const { return operator vfmav<T>(); }

    using parent::operator();
    template<typename... Ns> T &operator()(Ns... ns) const
      { return const_cast<T &>(parent::operator()(ns...)); }

    template<size_t nd2> vmav<T,nd2> subarray(const vector<slice> &slices) const
      {
      auto [ninfo, nofs] = tinfo::template subdata<nd2> (slices);
      return vmav<T,nd2> (ninfo, data()+nofs, *this);
      }

    T *data() const
     { return const_cast<T *>(tbuf::d); }
    // read access to element #i
    template<typename I> T &raw(I i) const
      { return data()[i]; }

    static vmav build_empty()
      {
      shape_t nshp;
      nshp.fill(0);
      return vmav(static_cast<T *>(nullptr), nshp);
      }

    static vmav build_noncritical(const shape_t &shape)
      {
      auto shape2 = noncritical_shape(shape, sizeof(T));
      vmav tmp(shape2);
      vector<slice> slc(ndim);
      for (size_t i=0; i<ndim; ++i) slc[i] = slice(0, shape[i]);
      return tmp.subarray<ndim>(slc);
      }
    static vmav build_noncritical(const shape_t &shape, uninitialized_dummy)
      {
      if (ndim<=1) return vmav(shape, UNINITIALIZED);
      auto shape2 = noncritical_shape(shape, sizeof(T));
      vmav tmp(shape2, UNINITIALIZED);
      vector<slice> slc(ndim);
      for (size_t i=0; i<ndim; ++i) slc[i] = slice(0, shape[i]);
      return tmp.subarray<ndim>(slc);
      }
    vmav transpose() const
      {
      return vmav(static_cast<const tinfo *>(this)->transpose(), *static_cast<const tbuf *>(this));
      }
    vmav<T, ndim+1> prepend_1() const
      {
      return vmav<T, ndim+1>(static_cast<const tinfo *>(this)->prepend_1(), *static_cast<const tbuf *>(this));
      }
    template<size_t ndim2> vmav<T, ndim2> reinterpret
      (const typename vmav<T, ndim2>::shape_t &newshp,
       const typename vmav<T, ndim2>::stride_t &newstr) const
      {
      return vmav<T, ndim2>(*static_cast<const tbuf *>(this), newshp, newstr);
      }
  };

template<size_t nd2, typename T, size_t ndim> vmav<T,nd2> subarray
  (const vmav<T, ndim> &arr, const vector<slice> &slices)  
  { return arr.template subarray<nd2>(slices); }

// various operations involving fmav objects of the same shape -- experimental

DUCC0_NOINLINE tuple<fmav_info::shape_t, vector<fmav_info::stride_t>, size_t, size_t>
  multiprep(const vector<fmav_info> &info, const vector<size_t> &tsizes);
DUCC0_NOINLINE tuple<fmav_info::shape_t, vector<fmav_info::stride_t>>
  multiprep(const vector<fmav_info> &info);

template<typename Ttuple> constexpr inline size_t tuplelike_size()
  { return tuple_size_v<remove_reference_t<Ttuple>>; }

template <typename Func, typename Ttuple, size_t... I>
inline void call_with_tuple_impl(Func &&func, const Ttuple& tuple,
  index_sequence<I...>)
  { func(std::forward<typename tuple_element<I, Ttuple>::type>(get<I>(tuple))...); }
template<typename Func, typename Ttuple> inline void call_with_tuple
  (Func &&func, Ttuple &&tuple)
  {
  call_with_tuple_impl(std::forward<Func>(func), tuple,
                       make_index_sequence<tuplelike_size<Ttuple>()>());
  }
template <typename Func, typename Ttuple, size_t... I>
inline void call_with_tuple2_impl(Func &&func, const Ttuple& tuple,
  index_sequence<I...>)
  { func(get<I>(tuple)...); }
template<typename Func, typename Ttuple> inline void call_with_tuple2
  (Func &&func, Ttuple &&tuple)
  {
  call_with_tuple2_impl(std::forward<Func>(func), tuple,
                        make_index_sequence<tuplelike_size<Ttuple>()>());
  }

template<typename...Ts, typename Func, size_t... Is>
inline auto tuple_transform_impl(tuple<Ts...> const& inputs, Func &&func,
  index_sequence<Is...>)
  { return tuple<invoke_result_t<Func,Ts>...>{func(get<Is>(inputs))...}; }
template<typename... Ts, typename Func>
inline auto tuple_transform(tuple<Ts...> const& inputs, Func &&func)
  {
  return tuple_transform_impl(inputs, std::forward<Func>(func),
                              make_index_sequence<sizeof...(Ts)>{});
  }
template<typename...Ts, typename Func, size_t... Is>
inline void tuple_for_each_impl(tuple<Ts...> &tpl, Func &&func,
  index_sequence<Is...>)
  { (func(get<Is>(tpl)), ...); }
template<typename... Ts, typename Func>
inline void tuple_for_each(tuple<Ts...> &tpl, Func &&func)
  {
  tuple_for_each_impl(tpl, std::forward<Func>(func), make_index_sequence<sizeof...(Ts)>{});
  }
template<typename...Ts, typename Func, size_t... Is>
inline void tuple_for_each_impl(const tuple<Ts...> &tpl, Func &&func,
  index_sequence<Is...>)
  { (func(get<Is>(tpl)), ...); }
template<typename... Ts, typename Func>
inline void tuple_for_each(const tuple<Ts...> &tpl, Func &&func)
  {
  tuple_for_each_impl(tpl, std::forward<Func>(func), make_index_sequence<sizeof...(Ts)>{});
  }

template<typename...Ts, typename Func, size_t... Is>
inline auto tuple_transform_idx_impl(const tuple<Ts...> &inputs,
   Func &&func, index_sequence<Is...>)
  {
  return tuple<invoke_result_t<Func, Ts, int>...>
    {func(get<Is>(inputs), Is)...};
  }

template<typename... Ts, typename Func>
inline auto tuple_transform_idx(const tuple<Ts...> &inputs, Func &&func)
  {
  return tuple_transform_idx_impl(inputs, std::forward<Func>(func),
                                  make_index_sequence<sizeof...(Ts)>{});
  }
template<typename...Ts, typename Func, size_t... Is>
inline void tuple_for_each_idx_impl(tuple<Ts...> &tpl, Func &&func,
  index_sequence<Is...>)
  { (func(get<Is>(tpl), Is), ...); }
template<typename... Ts, typename Func>
inline void tuple_for_each_idx(tuple<Ts...> &tpl, Func &&func)
  {
  tuple_for_each_idx_impl(tpl, std::forward<Func>(func), make_index_sequence<sizeof...(Ts)>{});
  }

template<typename Ttuple> inline auto to_ref (const Ttuple &tuple)
  {
  return tuple_transform(tuple,[](auto &&ptr) -> typename std::add_lvalue_reference_t<decltype(*ptr)>{ return *ptr; });
  }

template<typename Ttuple> inline Ttuple update_pointers (const Ttuple &ptrs,
  const vector<vector<ptrdiff_t>> &str, size_t idim, size_t i)
  {
  return tuple_transform_idx(ptrs, [i,idim,&str](auto &&ptr, size_t idx)
                             { return ptr + i*str[idx][idim]; });
  }

template<typename Ttuple> inline Ttuple update_pointers_contiguous (const Ttuple &ptrs,
  size_t i)
  {
  return tuple_transform(ptrs, [i](auto &&ptr) { return ptr+i; });
  }
template<typename Ttuple> inline void advance_contiguous (Ttuple &ptrs)
  { tuple_for_each(ptrs, [](auto &&ptr) { ++ptr; }); }
template<typename Ttuple> inline void advance (Ttuple &ptrs,
  const vector<vector<ptrdiff_t>> &str, size_t idim)
  {
  tuple_for_each_idx(ptrs, [idim,&str](auto &&ptr, size_t idx)
                     { ptr += str[idx][idim]; });
  }
template<typename Ttuple> inline void advance_by_n (Ttuple &ptrs,
  const vector<vector<ptrdiff_t>> &str, size_t idim, size_t n)
  {
  tuple_for_each_idx(ptrs, [idim,n,&str](auto &&ptr, size_t idx)
                     { ptr += n*str[idx][idim]; });
  }

template<typename Ttuple, typename Func>
  DUCC0_NOINLINE void applyHelper_block(size_t idim, const vector<size_t> &shp,
    const vector<vector<ptrdiff_t>> &str, size_t bsi, size_t bsj,
    const Ttuple &ptrs, Func &&func)
  {
  auto leni=shp[idim], lenj=shp[idim+1];
  size_t nbi = (leni+bsi-1)/bsi;
  size_t nbj = (lenj+bsj-1)/bsj;
  for (size_t bi=0; bi<nbi; ++bi)
    for (size_t bj=0; bj<nbj; ++bj)
      {
      auto locptrs(ptrs);
      advance_by_n(locptrs, str, idim, bi*bsi);
      advance_by_n(locptrs, str, idim+1, bj*bsj);
      for (size_t i=bi*bsi; i<min(leni, (bi+1)*bsi); ++i, advance(locptrs, str, idim))
        {
        auto locptrs2(locptrs);
        for (size_t j=bj*bsj; j<min(lenj, (bj+1)*bsj); ++j, advance(locptrs2, str, idim+1))
          call_with_tuple(func, to_ref(locptrs2));
        }
      }
  }

template<typename Ttuple, typename Func>
  DUCC0_NOINLINE void applyHelper(size_t idim, const vector<size_t> &shp,
    const vector<vector<ptrdiff_t>> &str, size_t block0, size_t block1,
    const Ttuple &ptrs, Func &&func, bool last_contiguous)
  {
  auto len = shp[idim];
  if ((idim+2==shp.size()) && (block0!=0))  // we should do blocking
    applyHelper_block(idim, shp, str, block0, block1, ptrs, func);
  else if (idim+1<shp.size())
    for (size_t i=0; i<len; ++i)
      applyHelper(idim+1, shp, str, block0, block1, update_pointers(ptrs, str, idim, i),
        func, last_contiguous);
  else
    {
    auto locptrs(ptrs);
    if (last_contiguous)
      for (size_t i=0; i<len; ++i, advance_contiguous(locptrs))
        call_with_tuple(func, to_ref(locptrs));
    else
      for (size_t i=0; i<len; ++i, advance(locptrs, str, idim))
        call_with_tuple(func, to_ref(locptrs));
    }
  }
template<typename Func, typename Ttuple>
  inline void applyHelper(const vector<size_t> &shp,
    const vector<vector<ptrdiff_t>> &str, size_t block0, size_t block1,
    const Ttuple &ptrs, Func &&func, size_t nthreads, bool last_contiguous)
  {
  if (shp.size()==0)
    call_with_tuple(std::forward<Func>(func), to_ref(ptrs));
  else if (nthreads==1)
    applyHelper(0, shp, str, block0, block1, ptrs, std::forward<Func>(func), last_contiguous);
  else
    execParallel(shp[0], nthreads, [&](size_t lo, size_t hi)
      {
      auto locptrs = update_pointers(ptrs, str, 0, lo);
      auto locshp(shp);
      locshp[0] = hi-lo;
      applyHelper(0, locshp, str, block0, block1, locptrs, func, last_contiguous);
      });
  }

template<typename Func, typename... Targs>
  void mav_apply(Func &&func, int nthreads, Targs... args)
  {
  vector<fmav_info> infos;
  (infos.push_back(args), ...);
  vector<size_t> tsizes;
  (tsizes.push_back(sizeof(args.data()[0])), ...);
  auto [shp, str, block0, block1] = multiprep(infos, tsizes);
  bool last_contiguous = true;
  if (shp.size()>0)
    for (const auto &s:str)
      last_contiguous &= (s.back()==1);

  auto ptrs = tuple_transform(forward_as_tuple(args...),
    [](auto &&arg){return arg.data();});
  applyHelper(shp, str, block0, block1, ptrs, std::forward<Func>(func), nthreads, last_contiguous);
  }

DUCC0_NOINLINE tuple<fmav_info::shape_t, vector<fmav_info::stride_t>>
  multiprep_noopt(const vector<fmav_info> &info);

template <typename Func, typename Arg, typename Ttuple, size_t... I>
inline void call_with_tuple_arg_impl(Func &&func, Arg &&arg, const Ttuple& tuple,
  index_sequence<I...>)
  { func(std::forward<typename tuple_element<I, Ttuple>::type>(get<I>(tuple))..., arg); }
template<typename Func, typename Arg, typename Ttuple> inline void call_with_tuple_arg
  (Func &&func, Arg &&arg, Ttuple &&tuple)
  {
  call_with_tuple_arg_impl(std::forward<Func>(func), arg, tuple,
                       make_index_sequence<tuplelike_size<Ttuple>()>());
  }
template<typename Ttuple, typename Func>
  DUCC0_NOINLINE void applyHelper_with_index(size_t idim, const vector<size_t> &shp,
    const vector<vector<ptrdiff_t>> &str, const Ttuple &ptrs, Func &&func,
    vector<size_t> &index)
  {
  auto len = shp[idim];
  if (idim+1<shp.size())
    {
    auto idxbak = index[idim];
    for (size_t i=0; i<len; ++i, ++index[idim])
      applyHelper_with_index(idim+1, shp, str, update_pointers(ptrs, str, idim, i),
        func, index);
    index[idim] = idxbak;
    }
  else
    {
    auto locptrs(ptrs);
    auto idxbak = index[idim];
    for (size_t i=0; i<len; ++i, ++index[idim], advance(locptrs, str, idim))
      call_with_tuple_arg(func, const_cast<const vector<size_t> &>(index), to_ref(locptrs));
    index[idim] = idxbak;
    }
  }
template<typename Func, typename Ttuple>
  inline void applyHelper_with_index(const vector<size_t> &shp,
    const vector<vector<ptrdiff_t>> &str, const Ttuple &ptrs, Func &&func,
    size_t nthreads, vector<size_t> &index)
  {
  if (shp.size()==0)
    call_with_tuple_arg(std::forward<Func>(func), const_cast<const vector<size_t> &>(index), to_ref(ptrs));
  else if (nthreads==1)
    applyHelper_with_index(0, shp, str, ptrs, std::forward<Func>(func), index);
  else
    execParallel(shp[0], nthreads, [&](size_t lo, size_t hi)
      {
      auto locptrs = update_pointers(ptrs, str, 0, lo);
      auto locshp(shp);
      locshp[0] = hi-lo;
      auto locidx(index);
      locidx[0]=lo;
      applyHelper_with_index(0, locshp, str, locptrs, func, locidx);
      });
  }
template<typename Func, typename... Targs>
  void mav_apply_with_index(Func &&func, int nthreads, Targs... args)
  {
  vector<fmav_info> infos;
  (infos.push_back(args), ...);
  auto [shp, str] = multiprep_noopt(infos);
  vector<size_t> index(shp.size(), 0);

  auto ptrs = tuple_transform(forward_as_tuple(args...),
    [](auto &&arg){return arg.data();});
  applyHelper_with_index(shp, str, ptrs, std::forward<Func>(func), nthreads, index);
  }


template<typename T, size_t ndim> class mavref
  {
  private:
    const mav_info<ndim> &info;
    T *d;

  public:
    using shape_t = typename mav_info<ndim>::shape_t;
    using stride_t = typename mav_info<ndim>::stride_t;
    mavref(const mav_info<ndim> &info_, T *d_) : info(info_), d(d_) {}
    template<typename... Ns> T &operator()(Ns... ns) const
      { return d[info.idx(ns...)]; }
    /// Returns the total number of entries in the object.
    size_t size() const { return info.size(); }
    /// Returns the shape of the object.
    const shape_t &shape() const { return info.shape(); }
    /// Returns the length along dimension \a i.
    size_t shape(size_t i) const { return info.shape(i); }
    /// Returns the strides of the object.
    const stride_t &stride() const { return info.stride(); }
    /// Returns the stride along dimension \a i.
    const ptrdiff_t &stride(size_t i) const { return info.stride(i); }
    /// Returns true iff the last dimension has stride 1.
    /**  Typically used for optimization purposes. */
    bool last_contiguous() const
      { return info.last_contiguous(); }
    /** Returns true iff the object is C-contiguous, i.e. if the stride of the
     *  last dimension is 1, the stride for the next-to-last dimension is the
     *  shape of the last dimension etc. */
    bool contiguous() const
      { return info.contiguous(); }
    /// Returns true iff this->shape and \a other.shape match.
    bool conformable(const mavref &other) const
      { return shape()==other.shape(); }
  };

template<typename T, size_t ndim>
  mavref<T, ndim> make_mavref(const mav_info<ndim> &info_, T *d_)
  { return mavref<T, ndim>(info_, d_); }

template<typename...Ts, typename ...Qs, typename Func, size_t... Is>
inline auto tuple_transform2_impl(const tuple<Ts...> &i1, const tuple<Qs...> &i2,
  Func &&func, index_sequence<Is...>)
  { return tuple<invoke_result_t<Func, Ts, Qs>...>{func(get<Is>(i1),get<Is>(i2))...}; }
template<typename... Ts, typename ...Qs, typename Func>
inline auto tuple_transform2(const tuple<Ts...> &i1, const tuple<Qs...> &i2,
  Func &&func)
  {
  return tuple_transform2_impl(i1, i2, std::forward<Func>(func),
                               make_index_sequence<sizeof...(Ts)>{});
  }
template<typename Tptrs, typename Tinfos>
  auto make_mavrefs(const Tptrs &ptrs, const Tinfos &infos)
  {
  return tuple_transform2(ptrs, infos, [](auto &&ptr, auto &&info)
    { return make_mavref(info, ptr); });
  }

template<size_t ndim> auto make_infos(const fmav_info &info)
  {
  if constexpr(ndim>0)
    MR_assert(ndim<=info.ndim(), "bad dimensionality");
  auto iterdim = info.ndim()-ndim;
  fmav_info fout({info.shape().begin(),info.shape().begin()+iterdim},
                 {info.stride().begin(),info.stride().begin()+iterdim});

  typename mav_info<ndim>::shape_t shp;
  typename mav_info<ndim>::stride_t str;
  if constexpr (ndim>0)  // just to silence compiler warnings
    for (size_t i=0; i<ndim; ++i)
      {
      shp[i] = info.shape(iterdim+i);
      str[i] = info.stride(iterdim+i);
      }
  mav_info<ndim> iout(shp, str);
  return make_tuple(fout, iout);
  }

template<typename Tptrs, typename Tinfos, typename Func>
  DUCC0_NOINLINE void flexible_mav_applyHelper(size_t idim, const vector<size_t> &shp,
    const vector<vector<ptrdiff_t>> &str, const Tptrs &ptrs,
    const Tinfos &infos, Func &&func)
  {
  auto len = shp[idim];
  auto locptrs(ptrs);
  if (idim+1<shp.size())
    for (size_t i=0; i<len; ++i, advance(locptrs, str, idim))
      flexible_mav_applyHelper(idim+1, shp, str, locptrs, infos, func);
  else
    for (size_t i=0; i<len; ++i, advance(locptrs, str, idim))
      call_with_tuple2(func, make_mavrefs(locptrs, infos));
  }
template<typename Tptrs, typename Tinfos, typename Func>
  DUCC0_NOINLINE void flexible_mav_applyHelper(const vector<size_t> &shp,
    const vector<vector<ptrdiff_t>> &str, const Tptrs &ptrs,
    const Tinfos &infos, Func &&func, size_t nthreads)
  {
  if (shp.size()==0)
    call_with_tuple2(func, make_mavrefs(ptrs, infos));
  else if (nthreads==1)
    flexible_mav_applyHelper(0, shp, str, ptrs, infos, std::forward<Func>(func));
  else
    execParallel(shp[0], nthreads, [&](size_t lo, size_t hi)
      {
      auto locptrs = update_pointers(ptrs, str, 0, lo);
      auto locshp(shp);
      locshp[0] = hi-lo;
      flexible_mav_applyHelper(0, locshp, str, locptrs, infos, func);
      });
  }

template<size_t ndim> struct Xdim { static constexpr size_t dim=ndim; };

template<typename Ttuple, typename Tdim, typename Func>
  void xflexible_mav_apply(const Ttuple &tuple, const Tdim &dim, Func &&func, size_t nthreads)
  {
  auto fullinfos = tuple_transform2(tuple, dim, [](const auto &arg, const auto &dim)
                                    { return make_infos<remove_reference_t<decltype(dim)>::dim>(fmav_info(arg)); });
  vector<fmav_info> iter_infos;
  tuple_for_each(fullinfos,[&iter_infos](const auto &entry){iter_infos.push_back(get<0>(entry));});
  auto [shp, str] = multiprep(iter_infos);

  auto infos2 = tuple_transform(fullinfos, [](const auto &arg)
                                { return get<1>(arg); });
  auto ptrs = tuple_transform(tuple, [](auto &&arg){return arg.data();});
  flexible_mav_applyHelper(shp, str, ptrs, infos2, std::forward<Func>(func), nthreads);
  }

template<size_t nd0, typename T0, typename Func>
  void flexible_mav_apply(Func &&func, size_t nthreads, T0 &&m0)
  {
  xflexible_mav_apply(forward_as_tuple(m0),
                      forward_as_tuple(Xdim<nd0>()),
                      std::forward<Func>(func), nthreads); 
  }

template<size_t nd0, size_t nd1, typename T0, typename T1, typename Func>
  void flexible_mav_apply(Func &&func, size_t nthreads, T0 &&m0, T1 &&m1)
  {
  xflexible_mav_apply(forward_as_tuple(m0, m1),
                      forward_as_tuple(Xdim<nd0>(), Xdim<nd1>()),
                      std::forward<Func>(func), nthreads); 
  }

template<size_t nd0, size_t nd1, size_t nd2,
         typename T0, typename T1, typename T2, typename Func>
  void flexible_mav_apply(Func &&func, size_t nthreads, T0 &&m0, T1 &&m1, T2 &&m2)
  {
  xflexible_mav_apply(forward_as_tuple(m0, m1, m2),
                      forward_as_tuple(Xdim<nd0>(), Xdim<nd1>(), Xdim<nd2>()),
                      std::forward<Func>(func), nthreads); 
  }

}

using detail_mav::UNINITIALIZED;
using detail_mav::fmav_info;
using detail_mav::mav_info;
using detail_mav::slice;
using detail_mav::MAXIDX;
using detail_mav::cfmav;
using detail_mav::vfmav;
using detail_mav::cmav;
using detail_mav::vmav;
using detail_mav::subarray;
using detail_mav::mav_apply;
using detail_mav::mav_apply_with_index;
using detail_mav::flexible_mav_apply;
}

#endif
