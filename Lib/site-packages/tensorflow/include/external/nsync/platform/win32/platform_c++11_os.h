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

#ifndef NSYNC_PLATFORM_WIN32_PLATFORM_CPP_OS_H_
#define NSYNC_PLATFORM_WIN32_PLATFORM_CPP_OS_H_

/* Windows doesn't implement clock_gettime(). */
#include <sys/timeb.h>

#define clock_gettime nsync_clock_gettime

#define CLOCK_REALTIME 0
#define TIMER_ABSTIME 1

NSYNC_CPP_START_
typedef int clockid_t;
int clock_gettime (clockid_t clk_id, struct timespec *tp);
NSYNC_CPP_END_

/* Avoid deprecation/safety warning for tmpfile().
   This is used for the C++11 build.   The copy in
   win32/platform.h is used for the win32 "native" build. */
#define tmpfile nsync_tmpfile_wrapper
static inline FILE *nsync_tmpfile_wrapper (void) {
        FILE *fp;
        if (tmpfile_s (&fp) != 0) {
                fp = NULL;
        }
        return (fp);
}

/* Windows lacks pthread_key_t. */
#include "Windows.h"
#define sched_yield() Sleep(0)
NSYNC_C_START_
/* Can't use TlsAlloc() because we use pthread_key_create's destructor
   argument, so we implement it anew. */
#define pthread_key_t nsync_pthread_key_t
typedef int nsync_pthread_key_t;
int nsync_pthread_key_create (nsync_pthread_key_t *pkey, void (*dest) (void *));
int nsync_pthread_key_delete (nsync_pthread_key_t key);
void *nsync_pthread_getspecific (nsync_pthread_key_t key);
int nsync_pthread_setspecific (nsync_pthread_key_t key, void *value);
#define pthread_key_create nsync_pthread_key_create
#define pthread_key_delete nsync_pthread_key_delete
#define pthread_getspecific nsync_pthread_getspecific
#define pthread_setspecific nsync_pthread_setspecific
NSYNC_C_END_

#endif /*NSYNC_PLATFORM_WIN32_PLATFORM_CPP_OS_H_*/
