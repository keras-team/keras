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

#ifndef NSYNC_PUBLIC_NSYNC_MU_WAIT_H_
#define NSYNC_PUBLIC_NSYNC_MU_WAIT_H_

/* nsync_mu_wait() and nsync_mu_wait_with_deadline() can be used instead of condition
   variables.  In many straightforward situations they are of equivalent
   performance and are somewhat easier to use, because unlike condition
   variables, they do not require that the waits be placed in a loop, and they
   do not require explicit wakeup calls.  Example:

   Definitions:
      static nsync_mu mu = NSYNC_MU_INIT;
      static int i = 0;  // protected by mu
      // Condition for use with nsync_mu_wait().
      static int int_is_zero (const void *v) { return (*(const int *)v == 0); }

   Waiter:
      nsync_mu_lock (&mu);
      // Wait until i is zero.
      nsync_mu_wait (&mu, &int_is_zero, &i, NULL);
      // i is known to be zero here.
      // ...
      nsync_mu_unlock (&mu);


   Thread potentially making i zero:
      nsync_mu_lock (&mu);
      i--;
      // No need to signal that i may have become zero.  The unlock call below
      // will evaluate waiters' conditions to decide which to wake.
      nsync_mu_unlock (&mu);

   It is legal to use conditional critical sections and condition variables 
   on the same mutex.

                              --------------

   The implementation benefits from determining whether waiters are waiting for
   the same condition; it may then evaluate a condition once on behalf
   of several waiters.  Two waiters have equal condition if their "condition"
   pointers are equal, and either:
   - their "condition_arg" pointers are equal, or
   - "condition_arg_eq" is non-null and
     (*condition_arg_eq) (condition_arg0, condition_arg1) returns non-zero.
   *condition_arg_eq will not be invoked unless the "condition" pointers
   are equal, and the "condition_arg" pointers are unequal.

   If many waiters wait for distinct conditions simultaneously, condition
   variables may be faster.
 */

#include "nsync_cpp.h"
#include "nsync_mu.h"
#include "nsync_time.h"

NSYNC_CPP_START_

struct nsync_note_s_; /* forward declaration for an nsync_note */

/* Return when (*condition) (condition_arg) is true.  Perhaps unlock and relock
   *mu while blocked waiting for the condition to become true.  nsync_mu_wait()
   is equivalent to nsync_mu_wait_with_deadline() with
   abs_deadline==nsync_time_no_deadline, and cancel_note==NULL.

   Requires that *mu be held on entry.
   See nsync_mu_wait_with_deadline() for more details on *condition and
   *condition_arg_eq.  */
void nsync_mu_wait (nsync_mu *mu, int (*condition) (const void *condition_arg),
		    const void *condition_arg,
		    int (*condition_arg_eq) (const void *a, const void *b));

/* Return when at least one of:  (*condition) (condition_arg) is true, the
   deadline expires, or *cancel_note is notified.  Perhaps unlock and relock *mu
   while blocked waiting for one of these events, but always return with *mu
   held.  Return 0 iff the (*condition) (condition_arg) is true on return, and
   otherwise either ETIMEDOUT or ECANCELED, depending on why the call returned
   early.  Callers should use abs_deadline==nsync_time_no_deadline for no
   deadline, and cancel_note==NULL for no cancellation.

   Requires that *mu be held on entry.

   The implementation may call *condition from any thread using the mutex, and 
   while holding *mu in either read or write mode; it guarantees that any
   thread calling *condition will hold *mu in some mode.
   Requires that (*condition) (condition_arg) neither modify state protected by
   *mu, nor return a value dependent on state not protected by *mu.  To depend
   on time, use the abs_deadline parameter.
   (Conventional use of condition variables have the same restrictions on the
   conditions tested by the while-loop.)
   If non-null, condition_arg_eq should return whether two condition_arg
   calls with the same "condition" pointer are considered equivalent; it should
   have no side-effects.  */
int nsync_mu_wait_with_deadline (nsync_mu *mu,
				 int (*condition) (const void *condition_arg),
				 const void *condition_arg,
				 int (*condition_arg_eq) (const void *a, const void *b),
				 nsync_time abs_deadline,
				 struct nsync_note_s_ *cancel_note);

/* Unlock *mu, which must be held in write mode, and wake waiters, if
   appropriate.  Unlike nsync_mu_unlock(), this call is not required to wake
   nsync_mu_wait/nsync_mu_wait_with_deadline calls on conditions that were
   false before this thread acquired the lock.  This call should be used only
   at the end of critical sections for which:
   - nsync_mu_wait and/or nsync_mu_wait_with_deadline are in use on the same
     mutex,
   - this critical section cannot make the condition true for any of those
     nsync_mu_wait/nsync_mu_wait_with_deadline waits, and
   - when performance is significantly improved by using this call. */
void nsync_mu_unlock_without_wakeup (nsync_mu *mu);

NSYNC_MU_WAIT_CPP_OVERLOAD_
NSYNC_CPP_END_

#endif /*NSYNC_PUBLIC_NSYNC_MU_WAIT_H_*/
