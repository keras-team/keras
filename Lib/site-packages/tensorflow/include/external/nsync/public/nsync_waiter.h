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

#ifndef NSYNC_PUBLIC_NSYNC_WAITER_H_
#define NSYNC_PUBLIC_NSYNC_WAITER_H_

/* nsync_wait_n() allows the client to wait on multiple objects (condition
   variables, nsync_notes, nsync_counters, etc.)  until at least one of them
   becomes ready, or a deadline expires.

   It can be thought of as rather like Unix's select() or poll(),
   except the the objects being waited for are synchronization
   data structures, rather than file descriptors.

   The client can construct new objects that can be waited for by implementing
   three routines.

   Examples:

   To wait on two nsync_notes n0, n1, and a nsync_counter c0, 
   with a deadline of abs_deadline:

	// Form an array of struct nsync_waitable_s, identifying the
	// objects and the corresponding descriptors. (static initialization
	// syntax is used for brevity)
	static struct nsync_waitable_s w[] = {
		{ &n0, &nsync_note_waitable_funcs },
		{ &n1, &nsync_note_waitable_funcs },
		{ &c0, &nsync_counter_waitable_funcs }
	};
	static struct nsync_waitable_s *pw[] = { &w[0], &w[1], &w[2] };
	int n = sizeof (w) / sizeof (w[0]);

	// Wait.  The mu, lock, and unlock arguments are NULL because
	// no condition variables are invovled. 
	int i = nsync_wait_n (NULL, NULL, NULL, abs_deadline, n, pw);
	if (i == n) {
		// timeout
	} else {
		// w[i].v became ready.
	}

    To wait on multiple condition variables, the mu/lock/unlock parameters are
    used.  Imagine cv0 and cv1 are signalled when predicates pred0() (under
    lock mu0) and pred1() (under lock mu1) become true respectively.  Assume
    that mu0 is acquired before mu1.
	static void lock2 (void *v) {  // lock two mutexes in order
		nsync_mu **mu = (nsync_mu **) v;
		nsync_mu_lock (mu[0]);
		nsync_mu_lock (mu[1]);
	}
	static void unlock2 (void *v) { // unlock two mutexes.
		nsync_mu **mu = (nsync_mu **) v;
		nsync_mu_unlock (mu[1]);
		nsync_mu_unlock (mu[0]);
	}

	// Describe the condition variables and the locks.
	static struct nsync_waitable_s w[] = {
		{ &cv0, &nsync_cv_waitable_funcs },
		{ &cv1, &nsync_cv_waitable_funcs }
	};
	static struct nsync_waitable_s *pw[] = { &w[0], &w[1] };
	nsync_mu *lock_list[] = { &mu0, &mu1 };
	int n = sizeof (w) / sizeof (w[0]);

	lock2 (list_list);
	while (!pred0 () && !pred1 ()) {
                // Wait for one of the condition variables to be signalled,
                // with no timeout.
		nsync_wait_n (lock_list, &lock2, &unlock2,
			      nsync_time_no_deadline, n, pw);
	}
	if (pred0 ()) { ... }
	if (pred1 ()) { ... }
	unlock2 (list_list);

   */

#include <inttypes.h>
#include <time.h>
#include "nsync_cpp.h"
#include "nsync_atomic.h"
#include "nsync_time.h"

NSYNC_CPP_START_

struct nsync_waitable_funcs_s;  /* forward declaration of struct that contains
                                   type dependent wait operations */

/* Clients wait on objects by forming an array of struct nsync_waitable_s.
   Each each element points to one object and its type-dependent functions. */
struct nsync_waitable_s {
	void *v;	/* pointer to object */
	/* pointer to type-dependent functions.  Use
		&nsync_note_waitable_funcs for an nsync_note,
		&nsync_counternote_waitable_funcs for an nsync_counter,
		&nsync_cv_waitable_funcs for an nsync_cv.  */
	const struct nsync_waitable_funcs_s *funcs;
};

/* Wait until at least one of *waitable[0,..,count-1] is has been notified, or
   abs_deadline is reached.  Return the index of the notified element of
   waitable[], or count if no such element exists.
   If mu!=NULL, (*unlock)(mu) is called after the thread is queued on the
   various waiters, and (*lock)(mu) is called before return; mu/lock/unlock are
   used to acquire and release the relevant locks whan waiting on condition
   variables.  */
int nsync_wait_n (void *mu, void (*lock) (void *), void (*unlock) (void *),
		  nsync_time abs_deadline, int count,
		  struct nsync_waitable_s *waitable[]);

/* --------------------------------------------------- */

/* A "struct nsync_waitable_s" implementation must implement these functions.
   Clients should ignore the internals. */
struct nsync_waiter_s;
struct nsync_waitable_funcs_s {
	/* Return the time when *v will be ready (max time if
	   unknown), or 0 if it is already ready.  The parameter nw may be
	   passed as NULL, in which case the result should indicate whether the
	   thread would block if it were to wait on *v. 
	   All calls with the same *v must report the same result until the
	   object becomes ready, from which point calls must report 0. */
	nsync_time (*ready_time) (void *v, struct nsync_waiter_s *nw);

	/* If *v is ready, return zero; otherwise enqueue *nw on *v and return
	   non-zero. */
	int (*enqueue) (void *v, struct nsync_waiter_s *nw);

        /* If nw has been previously dequeued, return zero; otherwise dequeue
           *nw from *v and return non-zero.  */
	int (*dequeue) (void *v, struct nsync_waiter_s *nw);
};

/* The "struct nsync_waitable_s" for nsync_note, nsync_counter, and nsync_cv. */
extern const struct nsync_waitable_funcs_s nsync_note_waitable_funcs;
extern const struct nsync_waitable_funcs_s nsync_counter_waitable_funcs;
extern const struct nsync_waitable_funcs_s nsync_cv_waitable_funcs;

NSYNC_WAITER_CPP_OVERLOAD_
NSYNC_CPP_END_

#endif /*NSYNC_PUBLIC_NSYNC_WAITER_H_*/
