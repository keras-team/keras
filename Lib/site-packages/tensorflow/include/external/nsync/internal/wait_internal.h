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

#ifndef NSYNC_INTERNAL_WAIT_INTERNAL_H_
#define NSYNC_INTERNAL_WAIT_INTERNAL_H_

NSYNC_CPP_START_

/* Implementations of "struct nsync_waitable_s" must provide functions in struct
   nsync_waitable_funcs_s (see public/nsync_wait.h).  When nsync_wait_n() waits
   on a client's object, those functions are called with v pointing to the
   client's object and nw pointing to a struct nsync_waiter_s. */
struct nsync_waiter_s {
	uint32_t tag; /* used for debugging */
	nsync_dll_element_ q; /* used to link children of parent */
	nsync_atomic_uint32_ waiting; /* non-zero <=> the waiter is waiting */
	struct nsync_semaphore_s_ *sem; /* *sem will be Ved when waiter is woken */
	uint32_t flags; /* see below */
};

#define NSYNC_WAITER_FLAG_MUCV 0x1 /* set if waiter is embedded in Mu/CV's internal structures */

NSYNC_CPP_END_

#endif /*NSYNC_INTERNAL_WAIT_INTERNAL_H_*/
