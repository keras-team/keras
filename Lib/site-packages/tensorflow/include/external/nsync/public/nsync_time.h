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

#ifndef NSYNC_PUBLIC_NSYNC_TIME_H_
#define NSYNC_PUBLIC_NSYNC_TIME_H_

#include "nsync_cpp.h"
#include "nsync_time_internal.h"

/* The type nsync_time represents the interval elapsed between two moments in
   time.  Often the first such moment is an address-space-wide epoch, such as
   the Unix epoch, but clients should not rely on the epoch in one address
   space being the same as that in another.  Intervals relative to the epoch
   are known as absolute times.

   The internals of nsync_time should be treated as opaque by clients.
   See nsync_time_internal.h. */

NSYNC_CPP_START_

extern const nsync_time nsync_time_no_deadline; /* A deadline infinitely far in the future. */
extern const nsync_time nsync_time_zero;  /* The zero delay, or an expired deadline. */

nsync_time nsync_time_now (void); /* Return the current time since the epoch.  */

/* Sleep for the specified delay.  Returns the unslept time
   which may be non-zero if the call was interrupted. */
nsync_time nsync_time_sleep (nsync_time delay);

/* Return a+b */
nsync_time nsync_time_add (nsync_time a, nsync_time b);

/* Return a-b */
nsync_time nsync_time_sub (nsync_time a, nsync_time b);

/*  Return +ve, 0, or -ve according to whether a>b, a==b, or a<b. */
int nsync_time_cmp (nsync_time a, nsync_time b);

/* Return the specified number of milliseconds as a time. */
nsync_time nsync_time_ms (unsigned ms);

/* Return the specified number of microseconds as a time. */
nsync_time nsync_time_us (unsigned us);

/* Return an nsync_time constructed from second and nanosecond components */
nsync_time nsync_time_s_ns (time_t s, unsigned ns);

NSYNC_CPP_END_

#endif /*NSYNC_PUBLIC_NSYNC_TIME_H_*/
