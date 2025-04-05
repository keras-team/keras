// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2016 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_CXX11_THREADPOOL_THREAD_YIELD_H
#define EIGEN_CXX11_THREADPOOL_THREAD_YIELD_H

// Try to come up with a portable way to yield
#define EIGEN_THREAD_YIELD() std::this_thread::yield()

#endif  // EIGEN_CXX11_THREADPOOL_THREAD_YIELD_H
