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

#ifndef NSYNC_PUBLIC_NSYNC_DEBUG_H_
#define NSYNC_PUBLIC_NSYNC_DEBUG_H_

/* Debugging operations for mutexes and condition variables.

   These operations should not be relied upon for normal functionality.  The
   implementation may be slow, output formats may change, and the
   implementation is free to yield the empty string.  */

#include "nsync_cpp.h"
#include "nsync_mu.h"
#include "nsync_cv.h"

NSYNC_CPP_START_

/* Place in buf[0,..,n-1] a nul-terminated, human readable string indicative of
   some of the internal state of the mutex or condition variable, and return
   buf.  If n>=4, buffer overflow is indicated by placing the characters "..."
   at the end of the string.

   The *_and_waiters() variants attempt to output the waiter lists in addition
   to the basic state.  These variants may acquire internal locks and follow
   internal pointers.  Thus, they are riskier if invoked in an address space
   whose overall health is uncertain.  */
char *nsync_mu_debug_state (nsync_mu *mu, char *buf, int n);
char *nsync_cv_debug_state (nsync_cv *cv, char *buf, int n);
char *nsync_mu_debug_state_and_waiters (nsync_mu *mu, char *buf, int n);
char *nsync_cv_debug_state_and_waiters (nsync_cv *cv, char *buf, int n);

/* Like nsync_*_debug_state_and_waiters(), but ignoring all locking and safety
   considerations, and using an internal, possibly static buffer that may be
   overwritten by subsequent or concurrent calls to these routines.  These
   variants should be used only from an interactive debugger, when all other
   threads are stopped; the debugger is expected to recover from errors.  */
char *nsync_mu_debugger (nsync_mu *mu);
char *nsync_cv_debugger (nsync_cv *cv);

NSYNC_CPP_END_

#endif /*NSYNC_PUBLIC_NSYNC_DEBUG_H_*/
