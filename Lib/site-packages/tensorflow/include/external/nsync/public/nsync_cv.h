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

#ifndef NSYNC_PUBLIC_NSYNC_CV_H_
#define NSYNC_PUBLIC_NSYNC_CV_H_

#include <inttypes.h>
#include "nsync_cpp.h"
#include "nsync_mu.h"
#include "nsync_atomic.h"
#include "nsync_time.h"

NSYNC_CPP_START_

struct nsync_dll_element_s_;
struct nsync_note_s_;

/* An nsync_cv is a condition variable in the style of Mesa, Java, POSIX, and Go's sync.Cond.
   It allows a thread to wait for a condition on state protected by a mutex,
   and to proceed with the mutex held and the condition true.

   See also nsync_mu_wait() and nsync_mu_wait_with_deadline(), which implement conditional
   critical sections.  In many cases, they are easier to use than condition
   variables.

   Usage:

   after making the desired predicate true, call:
       nsync_cv_signal (&cv); // If at most one thread can make use of the predicate becoming true.
   or
       nsync_cv_broadcast (&cv); // If multiple threads can make use of the predicate becoming true.

   To wait for a predicate with no deadline (assuming nsync_cv_broadcast() or
   nsync_cv_signal() is called whenever the predicate becomes true):
	nsync_mu_lock (&mu;)
	while (!some_predicate_protected_by_mu) { // the while-loop is required.
		nsync_cv_wait (&cv, &mu);
	}
	// predicate is now true
	nsync_mu_unlock (&mu);

   To wait for a predicate with a deadline (assuming nsync_cv_broadcast() or
   nsync_cv_signal() is called whenever the predicate becomes true):
	nsync_mu_lock (&mu);
	while (!some_predicate_protected_by_mu &&
	       nsync_cv_wait_with_deadline (&cv, &mu, abs_deadline, cancel_note) == 0) {
	}
	if (some_predicate_protected_by_mu) { // predicate is true
	} else { // predicate is false, and deadline expired, or cancel_note was notified.
	}
	nsync_mu_unlock (&mu);
   or, if the predicate is complex and you wish to write it just once and
   inline, you could use the following instead of the for-loop above:
	nsync_mu_lock (&mu);
	int pred_is_true = 0;
	int outcome = 0;
	while (!(pred_is_true = some_predicate_protected_by_mu) && outcome == 0) {
		outcome = nsync_cv_wait_with_deadline (&cv, &mu, abs_deadline, cancel_note);
	}
	if (pred_is_true) { // predicate is true
	} else { // predicate is false, and deadline expired, or cancel_note was notified.
	}
	nsync_mu_unlock (&mu);

   As the examples show, Mesa-style condition variables require that waits use
   a loop that tests the predicate anew after each wait.  It may be surprising
   that these are preferred over the precise wakeups offered by the condition
   variables in Hoare monitors.  Imprecise wakeups make more efficient use of
   the critical section, because threads can enter it while a woken thread is
   still emerging from the scheduler, which may take thousands of cycles.
   Further, they make the programme easier to read and debug by making the
   predicate explicit locally at the wait, where the predicate is about to be
   assumed; the reader does not have to infer the predicate by examining all
   the places where wakeups may occur. */
typedef struct nsync_cv_s_ {
	nsync_atomic_uint32_ word;  /* see bits below */
	struct nsync_dll_element_s_ *waiters; /* points to tail of list of waiters; under mu. */
} nsync_cv;

/* An nsync_cv should be zeroed to initialize, which can be accomplished by
   initializing with static initializer NSYNC_CV_INIT, or by setting the entire
   struct to 0, or using nsync_cv_init().  */
#define NSYNC_CV_INIT { NSYNC_ATOMIC_UINT32_INIT_, 0 }
void nsync_cv_init (nsync_cv *cv);

/* Wake at least one thread if any are currently blocked on *cv.  If
   the chosen thread is a reader on an nsync_mu, wake all readers and, if
   possible, a writer. */
void nsync_cv_signal (nsync_cv *cv);

/* Wake all threads currently blocked on *cv. */
void nsync_cv_broadcast (nsync_cv *cv);

/* Atomically release "mu" (which must be held on entry) and block the caller
   on *cv.  Wait until awakened by a call to nsync_cv_signal() or
   nsync_cv_broadcast(), or a spurious wakeup; then reacquire "mu", and return.
   Equivalent to a call to nsync_mu_wait_with_deadline() with
   abs_deadline==nsync_time_no_deadline, and cancel_note==NULL.  Callers should use
   nsync_cv_wait() in a loop, as with all standard Mesa-style condition
   variables.  See examples above.  */
void nsync_cv_wait (nsync_cv *cv, nsync_mu *mu);

/* Atomically release "mu" (which must be held on entry)
   and block the calling thread on *cv.  It then waits until awakened by a
   call to nsync_cv_signal() or nsync_cv_broadcast() (or a spurious wakeup), or by the time
   reaching abs_deadline, or by cancel_note being notified.  In all cases, it
   reacquires "mu", and returns the reason for the call returned (0, ETIMEDOUT,
   or ECANCELED).  Use abs_deadline==nsync_time_no_deadline for no deadline, and
   cancel_note==NULL for no cancellation.  wait_with_deadline() should be used in a
   loop, as with all Mesa-style condition variables.  See examples above.

   There are two reasons for using an absolute deadline, rather than a relative
   timeout---these are why pthread_cond_timedwait() also uses an absolute
   deadline.  First, condition variable waits have to be used in a loop; with
   an absolute times, the deadline does not have to be recomputed on each
   iteration.  Second, in most real programmes, some activity (such as an RPC
   to a server, or when guaranteeing response time in a UI), there is a
   deadline imposed by the specification or the caller/user; relative delays
   can shift arbitrarily with scheduling delays, and so after multiple waits
   might extend beyond the expected deadline.  Relative delays tend to be more
   convenient mostly in tests and trivial examples than they are in real
   programmes. */
int nsync_cv_wait_with_deadline (nsync_cv *cv, nsync_mu *mu,
                                 nsync_time abs_deadline,
				 struct nsync_note_s_ *cancel_note);

/* Like nsync_cv_wait_with_deadline(), but allow an arbitrary lock *v to be used,
   given its (*lock)(mu) and (*unlock)(mu) routines.  */
int nsync_cv_wait_with_deadline_generic (nsync_cv *cv,
				   void *mu, void (*lock) (void *), void (*unlock) (void *),
				   nsync_time abs_deadline,
				   struct nsync_note_s_ *cancel_note);

NSYNC_CV_CPP_OVERLOAD_
NSYNC_CPP_END_

#endif /*NSYNC_PUBLIC_NSYNC_CV_H_*/
