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

#ifndef NSYNC_PUBLIC_NSYNC_COUNTER_H_
#define NSYNC_PUBLIC_NSYNC_COUNTER_H_

#include <inttypes.h>
#include "nsync_cpp.h"
#include "nsync_mu.h"
#include "nsync_atomic.h"
#include "nsync_time.h"

NSYNC_CPP_START_

struct nsync_dll_element_s_;

/* An nsync_counter represents an unsigned integer that can count up and down,
   and wake waiters when zero.  */
typedef struct nsync_counter_s_ *nsync_counter;

/* Return a freshly allocated nsync_counter with the specified value,
   of NULL if an nsync_counter cannot be created.

   Any non-NULL returned value should be passed to nsync_counter_free() when no
   longer needed.  */
nsync_counter nsync_counter_new (uint32_t value);

/* Free resources associated with c.  Requires that c was allocated by
   nsync_counter_new(), and no concurrent or future operations are applied to
   c.  */
void nsync_counter_free (nsync_counter c);

/* Add delta to c, and return its new value.  It is a checkable runtime error
   to decrement c below 0, or to increment c (i.e., apply a delta > 0) after a
   waiter has waited.  */
uint32_t nsync_counter_add (nsync_counter c, int32_t delta);

/* Return the current value of c.  */
uint32_t nsync_counter_value (nsync_counter c);

/* Wait until c has value 0, or until abs_deadline, then return
   the value of c.  It is a checkable runtime error to increment c after
   a waiter may have been woken due to the counter reaching zero.
   If abs_deadline==nsync_time_no_deadline, the deadline
   is far in the future. */
uint32_t nsync_counter_wait (nsync_counter c, nsync_time abs_deadline);

NSYNC_COUNTER_CPP_OVERLOAD_
NSYNC_CPP_END_

#endif /*NSYNC_PUBLIC_NSYNC_COUNTER_H_*/
