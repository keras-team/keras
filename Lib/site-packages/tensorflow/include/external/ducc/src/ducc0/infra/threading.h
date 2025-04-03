/** \file ducc0/infra/threading.h
 *  Mulithreading support, similar to functionality provided by OpenMP
 *
 * \copyright Copyright (C) 2019-2023 Peter Bell, Max-Planck-Society
 * \authors Peter Bell, Martin Reinecke
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

#ifndef DUCC0_THREADING_H
#define DUCC0_THREADING_H

// Low level threading support can be influenced by the following macros:
// - DUCC0_NO_LOWLEVEL_THREADING: if defined, multithreading is disabled
//   and all parallel regions will be executed sequentially
//   on the invoking thread.
// - DUCC0_CUSTOM_LOWLEVEL_THREADING: if defined, external definitions of
//   Mutex, UniqueLock, LockGuard, CondVar, set_active_pool(),
//   and get active_pool() must be supplied in "ducc0_custom_lowlevel_threading.h"
//   and the code will use those.
// Both macros must not be defined at the same time.
// If neither macro is defined, standard ducc0 multihreading will be active.

#if (defined(DUCC0_NO_LOWLEVEL_THREADING) && defined(DUCC0_CUSTOM_LOWLEVEL_THREADING))
static_assert(false, "DUCC0_NO_LOWLEVEL_THREADING and DUCC0_CUSTOMLOWLEVEL_THREADING must not be both defined");
#endif

#if defined(DUCC0_STDCXX_LOWLEVEL_THREADING)
static_assert(false, "DUCC0_STDCXX_LOWLEVEL_THREADING must not be defined externally");
#endif

#if ((!defined(DUCC0_NO_LOWLEVEL_THREADING)) && (!defined(DUCC0_CUSTOM_LOWLEVEL_THREADING)))
#define DUCC0_STDCXX_LOWLEVEL_THREADING
#endif

#include <cstddef>
#include <functional>
#include <optional>
#include <vector>

#include "ducc0/infra/error_handling.h"

// threading-specific headers
#ifdef DUCC0_STDCXX_LOWLEVEL_THREADING
#include <mutex>
#include <condition_variable>
#endif

#ifdef DUCC0_NO_LOWLEVEL_THREADING
// no headers needed
#endif

namespace ducc0 {
namespace detail_threading {

using std::size_t;

/// Abstract base class for minimalistic thread pool functionality
class thread_pool
  {
  public:
    virtual ~thread_pool() {}
    /// Returns the total number of threads managed by the pool
    virtual size_t nthreads() const = 0;
    /** "Normalizes" a requested number of threads. A useful convention could be
        return (nthreads_in==0) ? nthreads() : min(nthreads(), nthreads_in); */
    virtual void resize(size_t /*nthreads_new*/)
      { MR_fail("Resizing is not supported by this thread pool"); }
    virtual size_t adjust_nthreads(size_t nthreads_in) const = 0;
    virtual void submit(std::function<void()> work) = 0;
  };

}}

#ifdef DUCC0_CUSTOM_LOWLEVEL_THREADING
#include "ducc0_custom_lowlevel_threading.h"
#endif

namespace ducc0 {

namespace detail_threading {

thread_pool *set_active_pool(thread_pool *new_pool);
thread_pool *get_active_pool();

// define threading related types dependent on the underlying implementation
#ifdef DUCC0_STDCXX_LOWLEVEL_THREADING
using Mutex = std::mutex;
using UniqueLock = std::unique_lock<std::mutex>;
using LockGuard = std::lock_guard<std::mutex>;
using CondVar = std::condition_variable;
#endif

#ifdef DUCC0_NO_LOWLEVEL_THREADING
struct Mutex
  {
  void lock(){}
  void unlock(){}
  };
struct LockGuard
  {
  LockGuard(const Mutex &){}
  };
struct UniqueLock
  {
  UniqueLock(const Mutex &){}
  void lock() {}
  void unlock() {}
  };
struct CondVar
  {
  template<class Predicate>
    void wait(UniqueLock &, Predicate) {}
  void notify_one() noexcept {}
  void notify_all() noexcept {}
  };
#endif

using std::size_t;

class ScopedUseThreadPool
  {
  private:
    thread_pool *old_pool_;
  public:
    ScopedUseThreadPool(thread_pool &pool)
      { old_pool_ = set_active_pool(&pool); }
    ~ScopedUseThreadPool()
      { set_active_pool(old_pool_); }
  };

/// Index range describing a chunk of work inside a parallelized loop
struct Range
  {
  size_t lo, //< first index of the chunk
         hi; //< one-past-last index of the chunk
  Range() : lo(0), hi(0) {}
  Range(size_t lo_, size_t hi_) : lo(lo_), hi(hi_) {}
  /// Returns true iff the chunk is not empty
  operator bool() const { return hi>lo; }
  };

/// Class supplied to parallel regions, which allows them to determine their
/// work chunks.
class Scheduler
  {
  public:
    virtual ~Scheduler() {}
    /// Returns the number of threads working in this parallel region
    virtual size_t num_threads() const = 0;
    /// Returns the number of this thread, from the range 0 to num_threads()-1.
    virtual size_t thread_num() const = 0;
    /// Returns information about the next chunk of work.
    /// If this chunk is empty, the work on this thread is done.
    virtual Range getNext() = 0;
  };

size_t available_hardware_threads();
/** Returns the maximum number of threads that are supported by currently
    active thread pool. */
size_t thread_pool_size();
void resize_thread_pool(size_t nthreads_new);
size_t adjust_nthreads(size_t nthreads);

/// Execute \a func over \a nwork work items, on a single thread.
void execSingle(size_t nwork,
  std::function<void(Scheduler &)> func);
/// Execute \a func over \a nwork work items, on \a nthreads threads.
/** Chunks will have the size \a chunksize, except for the last one which
 *  may be smaller.
 *
 *  Chunks are statically assigned to threads at startup. */
void execStatic(size_t nwork, size_t nthreads, size_t chunksize,
  std::function<void(Scheduler &)> func);
/// Execute \a func over \a nwork work items, on \a nthreads threads.
/** Chunks will have the size \a chunksize, except for the last one which
 *  may be smaller.
 *
 *  Chunks are assigned dynamically to threads;whenever a thread is finished
 *  with its current chunk, it will obtain the next one from the list of
 *  remaining chunks. */
void execDynamic(size_t nwork, size_t nthreads, size_t chunksize,
  std::function<void(Scheduler &)> func);
void execGuided(size_t nwork, size_t nthreads, size_t chunksize_min,
  double fact_max, std::function<void(Scheduler &)> func);
/// Execute \a func on \a nthreads threads.
/** Work subdivision must be organized within \a func. */
void execParallel(size_t nthreads, std::function<void(Scheduler &)> func);
/// Execute \a func on \a nthreads threads, passing only the thread number.
/** Work subdivision must be organized within \a func. */
void execParallel(size_t nthreads, std::function<void(size_t)> func);
/// Execute \a func on work items [\a lo; \a hi[ over \a nthreads threads.
/** Work items are subdivided fairly among threads. */
void execParallel(size_t work_lo, size_t work_hi, size_t nthreads,
  std::function<void(size_t, size_t)> func);
/// Execute \a func on work items [0; \a nwork[ over \a nthreads threads.
/** Work items are subdivided fairly among threads. */
inline void execParallel(size_t nwork, size_t nthreads,
  std::function<void(size_t, size_t)> func)
  { execParallel(0, nwork, nthreads, func); }
/// Execute \a func on work items [\a lo; \a hi[ over \a nthreads threads.
/** The first argument to \a func is the thread number.
 *
 *  Work items are subdivided fairly among threads. */
void execParallel(size_t work_lo, size_t work_hi, size_t nthreads,
  std::function<void(size_t, size_t, size_t)> func);
/// Execute \a func on work items [0; \a nwork[ over \a nthreads threads.
/** The first argument to \a func is the thread number.
 *
 *  Work items are subdivided fairly among threads. */
inline void execParallel(size_t nwork, size_t nthreads,
  std::function<void(size_t, size_t, size_t)> func)
  { execParallel(0, nwork, nthreads, func); }

template<typename T> class Worklist
  {
  private:
    Mutex mtx;
    CondVar cv;
    size_t nworking{0};
    std::vector<T> items;

  public:
    Worklist(const std::vector<T> &items_)
      : items(items_) {}

    std::optional<T> get_item()
      {
      UniqueLock lck(mtx);
      if ((--nworking==0) && items.empty()) cv.notify_all();
      cv.wait(lck,[&](){return (!items.empty()) || (nworking==0);});
      if (!items.empty())
        {
        auto res = items.back();
        items.pop_back();
        ++nworking;
        return res;
        }
      else
        return {};
      }
    void startup()
      {
      LockGuard lck(mtx);
      ++nworking;
      }
    void put_item(const T &item)
      {
      LockGuard lck(mtx);
      items.push_back(item);
      cv.notify_one();
      }
  };
  
/// Execute \a func on work items in \a items over \a nthreads threads.
/** While processing a work item, \a func may submit further items to the list
 *  of work items. For this purpose, \a func must take a const T &
 *  (the work item to be processed) as well as a function which also takes
 *  a const T & (the insert function). Work items will be assigned whenever a
 *  thread becomes available. */
template<typename T, typename Func> auto execWorklist
  (size_t nthreads, const std::vector<T> &items, Func &&func)
  {
  Worklist<T> wl(items);
  execParallel(nthreads, [&wl, &func](auto &) {
    wl.startup();
    while(auto wrk=wl.get_item())
      func(wrk.value(), [&wl](const T &item){wl.put_item(item);});
    });
  }

} // end of namespace detail_threading

using detail_threading::Mutex;
using detail_threading::LockGuard;
using detail_threading::UniqueLock;
using detail_threading::CondVar;
using detail_threading::thread_pool;
using detail_threading::ScopedUseThreadPool;
using detail_threading::available_hardware_threads;
using detail_threading::thread_pool_size;
using detail_threading::resize_thread_pool;
using detail_threading::adjust_nthreads;
using detail_threading::Scheduler;
using detail_threading::execSingle;
using detail_threading::execStatic;
using detail_threading::execDynamic;
using detail_threading::execGuided;
using detail_threading::execParallel;
using detail_threading::execWorklist;

} // end of namespace ducc0

#endif
