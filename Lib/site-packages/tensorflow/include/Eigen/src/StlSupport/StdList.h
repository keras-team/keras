// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009 Hauke Heibel <hauke.heibel@googlemail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_STDLIST_H
#define EIGEN_STDLIST_H

#ifndef EIGEN_STDLIST_MODULE_H
#error "Please include Eigen/StdList instead of including this file directly."
#endif

#include "details.h"

/**
 * This section contains a convenience MACRO which allows an easy specialization of
 * std::list such that for data types with alignment issues the correct allocator
 * is used automatically.
 */
#define EIGEN_DEFINE_STL_LIST_SPECIALIZATION(...)                                               \
  namespace std {                                                                               \
  template <>                                                                                   \
  class list<__VA_ARGS__, std::allocator<__VA_ARGS__> >                                         \
      : public list<__VA_ARGS__, EIGEN_ALIGNED_ALLOCATOR<__VA_ARGS__> > {                       \
    typedef list<__VA_ARGS__, EIGEN_ALIGNED_ALLOCATOR<__VA_ARGS__> > list_base;                 \
                                                                                                \
   public:                                                                                      \
    typedef __VA_ARGS__ value_type;                                                             \
    typedef list_base::allocator_type allocator_type;                                           \
    typedef list_base::size_type size_type;                                                     \
    typedef list_base::iterator iterator;                                                       \
    explicit list(const allocator_type& a = allocator_type()) : list_base(a) {}                 \
    template <typename InputIterator>                                                           \
    list(InputIterator first, InputIterator last, const allocator_type& a = allocator_type())   \
        : list_base(first, last, a) {}                                                          \
    list(const list& c) : list_base(c) {}                                                       \
    explicit list(size_type num, const value_type& val = value_type()) : list_base(num, val) {} \
    list(iterator start_, iterator end_) : list_base(start_, end_) {}                           \
    list& operator=(const list& x) {                                                            \
      list_base::operator=(x);                                                                  \
      return *this;                                                                             \
    }                                                                                           \
  };                                                                                            \
  }

#endif  // EIGEN_STDLIST_H
