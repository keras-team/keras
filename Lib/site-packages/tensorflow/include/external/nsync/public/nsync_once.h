/* Copyright 2016 Google Inc.

  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License. */

#ifndef NSYNC_PUBLIC_NSYNC_ONCE_H_
#define NSYNC_PUBLIC_NSYNC_ONCE_H_

#include <inttypes.h>
#include "nsync_cpp.h"
#include "nsync_atomic.h"

NSYNC_CPP_START_

/* An nsync_once allows a function to be called exactly once, when first referenced. */
typedef nsync_atomic_uint32_ nsync_once;

/* An initializer for nsync_once; it is guaranteed to be all zeroes. */
#define NSYNC_ONCE_INIT NSYNC_ATOMIC_UINT32_INIT_

/* The first time nsync_run_once() or nsync_run_once_arg() is applied to *once,
   the supplied function is run (with argument, in the case of nsync_run_once_arg()).
   Other callers will wait until the run of the function is complete, and then
   return without running the function again. */
void nsync_run_once (nsync_once *once, void (*f) (void));
void nsync_run_once_arg (nsync_once *once, void (*farg) (void *arg), void *arg);

/* Same as nsync_run_once()/nsync_run_once_arg() but uses a spinloop.
   Can be used on the same nsync_once as nsync_run_once/nsync_run_once_arg().

   These *_spin variants should be used only in contexts where normal blocking
   is disallowed, such as within user-space schedulers, when the runtime is
   not fully initialized, etc.  They provide no significant performance benefit,
   and they should be avoided in normal code. */
void nsync_run_once_spin (nsync_once *once, void (*f) (void));
void nsync_run_once_arg_spin (nsync_once *once, void (*farg) (void *arg), void *arg);

NSYNC_CPP_END_

#endif /*NSYNC_PUBLIC_NSYNC_ONCE_H_*/
