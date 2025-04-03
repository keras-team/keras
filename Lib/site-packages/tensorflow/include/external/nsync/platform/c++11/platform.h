/* Copyright 2017 Google Inc.

  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License. */

#ifndef NSYNC_PLATFORM_CPP11_PLATFORM_H_
#define NSYNC_PLATFORM_CPP11_PLATFORM_H_

/* These C header files are in "C compatibility headers" in C++11. */
#include <string.h>
#include <errno.h>
#include <stdlib.h>
#include <stddef.h>
#include <inttypes.h>
#include <limits.h>

#include <stdio.h>
#include <stdarg.h>

#include <mutex>
#include <condition_variable>

#include "nsync_cpp.h"

/* We could include pthread.h, if we knew it existed.
   Instead we implement the things the tests need using C++11 equivalents. */

typedef std::mutex *nsync_pthread_mutex_;
static inline int nsync_pthread_mutex_lock_ (nsync_pthread_mutex_ *mu) { (*mu)->lock (); return (0); }
static inline int nsync_pthread_mutex_unlock_ (nsync_pthread_mutex_ *mu) { (*mu)->unlock (); return (0); }
#undef pthread_mutex_t
#undef pthread_mutex_lock
#undef pthread_mutex_unlock
#undef pthread_mutex_init
#undef pthread_mutex_destroy
#define pthread_mutex_t nsync_pthread_mutex_
#define pthread_mutex_lock nsync_pthread_mutex_lock_
#define pthread_mutex_unlock nsync_pthread_mutex_unlock_
#define pthread_mutex_init(mu,attr) (*(mu) = new std::mutex)
#define pthread_mutex_destroy(mu) delete *(mu)

typedef std::condition_variable *nsync_pthread_cond_;
static inline int nsync_pthread_cond_broadcast_ (nsync_pthread_cond_ *cv) { (*cv)->notify_all (); return (0); }
static inline int nsync_pthread_cond_signal_ (nsync_pthread_cond_ *cv) { (*cv)->notify_one (); return (0); }
static inline int nsync_pthread_cond_wait_ (nsync_pthread_cond_ *cv, nsync_pthread_mutex_ *mu) {
	std::unique_lock<std::mutex> mu_mu (**mu, std::adopt_lock);
	(*cv)->wait (mu_mu);
	mu_mu.release ();
	return (0);
}
static inline int nsync_pthread_cond_timedwait_ (nsync_pthread_cond_ *cv, nsync_pthread_mutex_ *mu,
						const struct timespec *abstimeout) {
	std::cv_status result = std::cv_status::no_timeout;
	std::unique_lock<std::mutex> mu_mu (**mu, std::adopt_lock);
	if (abstimeout == NULL || abstimeout->tv_sec >= 0x7fffffff) {
		(*cv)->wait (mu_mu);
	} else {
		std::chrono::system_clock::time_point epoch;
		result = (*cv)->wait_until (mu_mu,
					 epoch + std::chrono::nanoseconds (
					     abstimeout->tv_nsec +
					     1000 * 1000 * 1000 * (int64_t) abstimeout->tv_sec));
	}
	mu_mu.release ();
	return (result == std::cv_status::timeout? ETIMEDOUT : 0);
}
#undef pthread_cond_t
#undef pthread_cond_broadcast
#undef pthread_cond_signal
#undef pthread_cond_wait
#undef pthread_cond_timedwait
#undef pthread_cond_init
#undef pthread_cond_destroy
#define pthread_cond_t nsync_pthread_cond_
#define pthread_cond_broadcast nsync_pthread_cond_broadcast_
#define pthread_cond_signal nsync_pthread_cond_signal_
#define pthread_cond_wait nsync_pthread_cond_wait_
#define pthread_cond_timedwait nsync_pthread_cond_timedwait_
#define pthread_cond_init(cv, attr) (*(cv) = new std::condition_variable)
#define pthread_cond_destroy(cv) delete *(cv)

typedef struct nsync_pthread_rwlock_s {
	nsync_pthread_rwlock_s () : lock (0) {}
        int lock; /* -1 == exclusive; 0==unlocked; +ve == reader count */
        std::mutex mu;
        std::condition_variable cv;
} *nsync_pthread_rwlock_;
static inline int nsync_pthread_rwlock_wrlock_ (nsync_pthread_rwlock_ *rw) {
	std::unique_lock<std::mutex> rw_mu ((*rw)->mu, std::defer_lock);
	rw_mu.lock ();
	while ((*rw)->lock != 0) {
		(*rw)->cv.wait (rw_mu);
	}
	(*rw)->lock = -1;
	rw_mu.unlock ();
	return (0);
}
static inline int nsync_pthread_rwlock_rdlock_ (nsync_pthread_rwlock_ *rw) {
	std::unique_lock<std::mutex> rw_mu ((*rw)->mu, std::defer_lock);
	rw_mu.lock ();
	while ((*rw)->lock == -1) {
		(*rw)->cv.wait (rw_mu);
	}
	(*rw)->lock++;
	rw_mu.unlock ();
	return (0);
}
static inline int nsync_pthread_rwlock_unlock_ (nsync_pthread_rwlock_ *rw) {
	std::unique_lock<std::mutex> rw_mu ((*rw)->mu, std::defer_lock);
	rw_mu.lock ();
	if ((*rw)->lock == -1) {
		(*rw)->lock = 0;
	} else if ((*rw)->lock > 0) {
		(*rw)->lock--;
	}
	if ((*rw)->lock == 0) {
		(*rw)->cv.notify_all ();
	}
	rw_mu.unlock ();
	return (0);
}
#undef pthread_rwlock_t
#undef pthread_rwlock_wrlock
#undef pthread_rwlock_rdlock
#undef pthread_rwlock_unlock
#undef pthread_rwlock_init
#undef pthread_rwlock_destroy
#define pthread_rwlock_t nsync_pthread_rwlock_
#define pthread_rwlock_wrlock nsync_pthread_rwlock_wrlock_
#define pthread_rwlock_rdlock nsync_pthread_rwlock_rdlock_
#define pthread_rwlock_unlock nsync_pthread_rwlock_unlock_
#define pthread_rwlock_init(rw,attr) (*(rw) = new nsync_pthread_rwlock_s)
#define pthread_rwlock_destroy(rw) delete *(rw)


typedef int nsync_pthread_once_type_;
#undef PTHREAD_ONCE_INIT
#define PTHREAD_ONCE_INIT 0
static inline void nsync_pthread_once_ (nsync_pthread_once_type_ *o, void (*f) ()) {
	static std::mutex mu;
	static std::condition_variable cv;
	std::unique_lock<std::mutex> o_mu (mu, std::defer_lock);
	o_mu.lock ();
	if (*o == 0) {
		*o = 1;
		o_mu.unlock ();
		(*f) ();
		o_mu.lock ();
		*o = 2;
		cv.notify_all ();
	} else {
		while (*o != 2) {
			cv.wait (o_mu);
		}
	}
	o_mu.unlock ();
}
#undef pthread_once_t
#undef pthread_once
#define pthread_once_t nsync_pthread_once_type_
#define pthread_once nsync_pthread_once_

/* Some platforms need more OS-specific help. */
#include "platform_c++11_os.h"

#endif /*NSYNC_PLATFORM_CPP11_PLATFORM_H_*/
