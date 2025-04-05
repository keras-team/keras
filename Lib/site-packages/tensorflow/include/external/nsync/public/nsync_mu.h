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

#ifndef NSYNC_PUBLIC_NSYNC_MU_H_
#define NSYNC_PUBLIC_NSYNC_MU_H_

#include <inttypes.h>
#include "nsync_cpp.h"
#include "nsync_atomic.h"

NSYNC_CPP_START_

struct nsync_dll_element_s_;

/* An nsync_mu is a lock.  If initialized to all zeroes, it is valid and unlocked.

   An nsync_mu can be "free", held by a single thread (aka fiber, goroutine) in
   "write" (exclusive) mode, or by many threads in "read" (shared) mode.  A
   thread that acquires it should eventually release it.  It is illegal to
   acquire an nsync_mu in one thread and release it in another.  It is
   illegal for a thread to reacquire an nsync_mu while holding it (even a
   second share of a "read" lock).

   Example usage:
	static struct foo {
		nsync_mu mu; // protects invariant a+b==0 on fields below.
		int a;
		int b;
	} p = { NSYNC_MU_INIT, 0, 0 };
	....
	nsync_mu_lock (&p.mu);
	// The current thread now has exclusive access to p.a and p.b; invariant assumed true.
	p.a++;
	p.b--; // restore invariant p.a+p.b==0 before releasing p.mu
	nsync_mu_unlock (&p.mu)

   Mutexes can be used with condition variables; see nsync_cv.h.

   nsync_mu_wait() and nsync_mu_wait_with_deadline() can be used instead of
   condition variables.  See nsync_mu_wait.h for more details.
   Example use of nsync_mu_wait() to wait for p.a==0, using definition above:
	int a_is_zero (const void *condition_arg) {
		return (((const struct foo *)condition_arg)->a == 0);
	}
	...
	nsync_mu_lock (&p.mu);
	nsync_mu_wait (&p.mu, &a_is_zero, &p, NULL);
	// The current thread now has exclusive access to p.a and p.b, and p.a==0.
	...
	nsync_mu_unlock (&p.mu); */
typedef struct nsync_mu_s_ {
	nsync_atomic_uint32_ word; /* internal use only */
	struct nsync_dll_element_s_ *waiters; /* internal use only */
} nsync_mu;

/* An nsync_mu should be zeroed to initialize, which can be accomplished by
   initializing with static initializer NSYNC_MU_INIT, or by setting the entire
   structure to all zeroes, or using nsync_mu_init().  */
#define NSYNC_MU_INIT { NSYNC_ATOMIC_UINT32_INIT_, 0 }
void nsync_mu_init (nsync_mu *mu);

/* Block until *mu is free and then acquire it in writer mode.
   Requires that the calling thread not already hold *mu in any mode.  */
void nsync_mu_lock (nsync_mu *mu);

/* Unlock *mu, which must have been acquired in write mode by the calling
   thread, and wake waiters, if appropriate.  */
void nsync_mu_unlock (nsync_mu *mu);

/* Attempt to acquire *mu in writer mode without blocking, and return non-zero
   iff successful.  Return non-zero with high probability if *mu was free
   on entry.  */
int nsync_mu_trylock (nsync_mu *mu);

/* Block until *mu can be acquired in reader mode and then acquire it.
   Requires that the calling thread not already hold *mu in any mode. */
void nsync_mu_rlock (nsync_mu *mu);

/* Unlock *mu, which must have been acquired in read mode by the calling
   thread, and wake waiters, if appropriate.  */
void nsync_mu_runlock (nsync_mu *mu);

/* Attempt to acquire *mu in reader mode without blocking, and return non-zero
   iff successful.  Return non-zero with high probability if *mu was free on
   entry.  Perhaps fail to acquire if a writer is waiting, to avoid starvation.
   */
int nsync_mu_rtrylock (nsync_mu *mu);

/* May abort if *mu is not held in write mode by the calling thread. */
void nsync_mu_assert_held (const nsync_mu *mu);

/* May abort if *mu is not held in read or write mode
   by the calling thread.  */
void nsync_mu_rassert_held (const nsync_mu *mu);

/* Return whether *mu is held in read mode.
   Requires that the calling thread holds *mu in some mode. */
int nsync_mu_is_reader (const nsync_mu *mu);

NSYNC_CPP_END_

#endif /*NSYNC_PUBLIC_NSYNC_MU_H_*/
