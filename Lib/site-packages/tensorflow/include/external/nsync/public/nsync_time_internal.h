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

#ifndef NSYNC_PUBLIC_NSYNC_TIME_INTERNAL_H_
#define NSYNC_PUBLIC_NSYNC_TIME_INTERNAL_H_

#include "nsync_cpp.h"

/* Internal details of the implementation of the type nsync_time.

   The type nsync_time can have different implementations on different
   platforms, because the world has many different representations of time.
   Further, the "epoch" of absolute times can vary from address space to
   address space.

   On monotonic clocks:  In our testing, we found that the monotonic clock on
   various popular systems (such as Linux, and some BSD variants) was no better
   behaved than the realtime clock, and routinely took large steps backwards,
   especially on multiprocessors.  Given that "monotonic" doesn't seem to mean
   what it says, implementers of nsync_time might consider retaining the
   simplicity of a single epoch within an address space, by configuring any
   time synchronization mechanism (like ntp) to adjust for leap seconds by
   adjusting the rate, rather than with a backwards step.  */

#if NSYNC_USE_GPR_TIMESPEC
#include "grpc/support/time.h"
NSYNC_CPP_START_
typedef gpr_timespec nsync_time;
#define NSYNC_TIME_SEC(t) ((t).tv_sec)
#define NSYNC_TIME_NSEC(t) ((t).tv_nsec)
NSYNC_CPP_END_

#elif defined(NSYNC_USE_INT_TIME)
#include <time.h>
NSYNC_CPP_START_
typedef NSYNC_USE_INT_TIME nsync_time;
#define NSYNC_TIME_SEC(t)  (sizeof (nsync_time) >= 8? \
			    (t) / (1000 * 1000 * 1000): \
			    ((t) / 1000))
#define NSYNC_TIME_NSEC(t) (sizeof (nsync_time) >= 8? \
                            (t) % (1000 * 1000 * 1000): \
			    (((t) % 1000) * 1000 * 1000))
#define NSYNC_TIME_MAX_ MAX_INT_TYPE (nsync_time)
NSYNC_CPP_END_

#elif defined(NSYNC_USE_FLOATING_TIME)
#include <math.h>
#include <time.h>
NSYNC_CPP_START_
typedef NSYNC_USE_FLOATING_TIME nsync_time;
#define NSYNC_TIME_SEC(t)  (trunc ((t) / (nsync_time) (1000 * 1000 * 1000)))
#define NSYNC_TIME_NSEC(t) ((t) - ((1000 * 1000 * 1000) * NSYNC_TIME_SEC (t)))
#define NSYNC_TIME_MAX_ DBL_MAX
NSYNC_CPP_END_

#elif NSYNC_USE_DEBUG_TIME
/* Check that the library can be built with a different time struct.  */
#include <time.h>
NSYNC_CPP_START_
typedef struct {
	time_t seconds;
	unsigned nanoseconds;
} nsync_time;
#define NSYNC_TIME_SEC(t) ((t).seconds)
#define NSYNC_TIME_NSEC(t) ((t).nanoseconds)
NSYNC_CPP_END_

#elif defined(__cplusplus) && \
      (NSYNC_USE_CPP11_TIMEPOINT || (__cplusplus >= 201103L) || (_MSC_VER >= 1700))
/* The inline functions below provide function overloads that accept the most
   likely C++11 time type(s).

   C++11 time types have many variations and subtleties:
   - There are multiple clocks with potentially differing epochs; these clocks
     are not necessarily phase-locked to the same rate, making conversion and
     comparison between clocks tricky.
   - Relative and absolute times are distinguished in the type system.
   - Either integral or floating point counters may be used to represent time
     intervals, and code valid with one may not be valid with the other
     (see std::chrono::treat_as_floating_point).
   - A counter increment of one can represent any rational number of seconds
     (for whatever "seconds" means for this clock).
   - Conversions between duration types may round or truncate at the
     implementation's discretion.
   - As mentioned above, common implementations of the default monotonic clock
     ("steady_clock") illegally allow a thread to observe time going backwards,
     especially in the face of scheduling on a different CPU, making its use
     misleading, at best.
   I've chosen to handle this complexity by doing a conversion to absolute
   timespec at the interface layer, so all the C++ complication is here, rather
   than spread throughout the library.  */

#include <chrono>
#include <time.h>
NSYNC_CPP_START_
typedef struct timespec nsync_time;
#define NSYNC_TIME_SEC(t) ((t).tv_sec)
#define NSYNC_TIME_NSEC(t) ((t).tv_nsec)

typedef std::chrono::system_clock::time_point nsync_cpp_time_point_;
nsync_time nsync_from_time_point_ (nsync_cpp_time_point_);
nsync_cpp_time_point_ nsync_to_time_point_ (nsync_time);
#define NSYNC_COUNTER_CPP_OVERLOAD_ \
	static inline uint32_t nsync_counter_wait (nsync_counter c, \
						   nsync_cpp_time_point_ abs_deadline) { \
		return (nsync_counter_wait (c, nsync_from_time_point_ (abs_deadline))); \
	}
#define NSYNC_CV_CPP_OVERLOAD_ \
	static inline int nsync_cv_wait_with_deadline (nsync_cv *cv, nsync_mu *mu, \
		nsync_cpp_time_point_ abs_deadline, struct nsync_note_s_ *cancel_note) { \
		return (nsync_cv_wait_with_deadline (cv, mu, \
				nsync_from_time_point_ (abs_deadline), \
				cancel_note)); \
	} \
	static inline int nsync_cv_wait_with_deadline_generic (nsync_cv *cv, \
		void *mu, void (*lock) (void *), void (*unlock) (void *), \
		nsync_cpp_time_point_ abs_deadline, struct nsync_note_s_ *cancel_note) { \
		return (nsync_cv_wait_with_deadline_generic (cv, mu, lock, unlock, \
				nsync_from_time_point_ (abs_deadline), \
				cancel_note)); \
	}
#define NSYNC_MU_WAIT_CPP_OVERLOAD_ \
	static inline int nsync_mu_wait_with_deadline (nsync_mu *mu, \
		int (*condition) (const void *condition_arg), const void *condition_arg, \
		int (*condition_arg_eq) (const void *a, const void *b), \
		nsync_cpp_time_point_ abs_deadline, struct nsync_note_s_ *cancel_note) { \
		return (nsync_mu_wait_with_deadline (mu, condition, condition_arg, \
						     condition_arg_eq, \
						     nsync_from_time_point_ (abs_deadline), \
						     cancel_note)); \
	}
#define NSYNC_NOTE_CPP_OVERLOAD_ \
	static inline nsync_note nsync_note_new (nsync_note parent, \
						 nsync_cpp_time_point_ abs_deadline) { \
		return (nsync_note_new (parent, nsync_from_time_point_ (abs_deadline))); \
	} \
	static inline int nsync_note_wait (nsync_note n, nsync_cpp_time_point_ abs_deadline) { \
		return (nsync_note_wait (n, nsync_from_time_point_ (abs_deadline))); \
	} \
	static inline nsync_cpp_time_point_ nsync_note_expiry_timepoint (nsync_note n) { \
		return (nsync_to_time_point_ (nsync_note_expiry (n))); \
	}
#define NSYNC_WAITER_CPP_OVERLOAD_ \
	static inline int nsync_wait_n (void *mu, void (*lock) (void *), \
					void (*unlock) (void *), \
					nsync_cpp_time_point_ abs_deadline, \
					int count, struct nsync_waitable_s *waitable[]) { \
		return (nsync_wait_n (mu, lock, unlock, \
				      nsync_from_time_point_ (abs_deadline), count, waitable)); \
	}

NSYNC_CPP_END_

#else
/* Default is to use timespec. */
#include <time.h>
NSYNC_CPP_START_
typedef struct timespec nsync_time;
#define NSYNC_TIME_SEC(t) ((t).tv_sec)
#define NSYNC_TIME_NSEC(t) ((t).tv_nsec)
NSYNC_CPP_END_

#endif

#if !defined(NSYNC_COUNTER_CPP_OVERLOAD_)
#define NSYNC_COUNTER_CPP_OVERLOAD_
#define NSYNC_CV_CPP_OVERLOAD_
#define NSYNC_MU_WAIT_CPP_OVERLOAD_
#define NSYNC_NOTE_CPP_OVERLOAD_
#define NSYNC_WAITER_CPP_OVERLOAD_
#endif

#endif /*NSYNC_PUBLIC_NSYNC_TIME_INTERNAL_H_*/
