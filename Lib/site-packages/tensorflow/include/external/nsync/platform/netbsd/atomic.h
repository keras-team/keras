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

#ifndef NSYNC_PLATFORM_NETBSD_ATOMIC_H_
#define NSYNC_PLATFORM_NETBSD_ATOMIC_H_

/* Atomic operations on nsync_atomic_uint32_ quantities
   CAS, load, and store.

   Normally, these are used only on nsync_atomic_uint32_ values, but on Linux they may be
   invoked on int values, because futexes operate on int values.  A
   compile-time check in the futex code ensures that both int and   
   nsync_atomic_uint32_ are 32 bits.

   Memory barriers:
	   Operations with the suffixes _ACQ and _RELACQ ensure that the operation
	   appears to complete before other memory operations subsequently performed by
	   the same thread, as seen by other threads.  (In the case of ATM_CAS_ACQ,
	   this applies only if the operation returns a non-zero value.)

	   Operations with the suffixes _REL and _RELACQ ensure that the operation
	   appears to complete after other memory operations previously performed by
	   the same thread, as seen by other threads.  (In the case of ATM_CAS_REL,
	   this applies only if the operation returns a non-zero value.)

   // Atomically,
   //   int ATM_CAS (nsync_atomic_uint32_ *p, uint32_t old_value, uint32_t new_value) {
   //		if (*p == old_value) {
   //			*p = new_value;
   //			return (some-non-zero-value);
   //		} else {
   //			return (0);
   //		}
   //	}
   // *_ACQ, *_REL, *_RELACQ variants are available,
   // with the barrier semantics described above.
   int ATM_CAS (nsync_atomic_uint32_ *p, uint32_t old_value, uint32_t new_value);

   // Atomically,
   //     uint32_t ATM_LOAD (nsync_atomic_uint32_ *p) { return (*p); }
   // A *_ACQ variant is available,
   // with the barrier semantics described above.
   uint32_t ATM_LOAD (nsync_atomic_uint32_ *p);

   // Atomically,
   //     void ATM_STORE (nsync_atomic_uint32_ *p, uint32_t value) { *p = value; }
   // A *_REL variant is available,
   // with the barrier semantics described above.
   void ATM_STORE (nsync_atomic_uint32_ *p, uint32_t value);
 */

#include "compiler.h"
#include "nsync_atomic.h"
#include <sys/atomic.h>

NSYNC_CPP_START_

#if ATM_LD_IS_ACQ_ST_IS_REL_ && defined(__GNUC__)
#define ATM_CB_() __asm__ __volatile__ ("" : : : "memory")     /* compiler barrier */
#define ATM_LD_ACQ_ ATM_CB_
#define ATM_ST_REL_ ATM_CB_
#else
#define ATM_LD_ACQ_ membar_sync   /* not membar_enter(); it doesn't affect preceding loads */
#define ATM_ST_REL_ membar_exit
#endif

static INLINE int atm_cas_nomb_u32_ (nsync_atomic_uint32_ *p, uint32_t o, uint32_t n) {
	return (atomic_cas_32 (NSYNC_ATOMIC_UINT32_PTR_ (p), o, n) == o);
}

static INLINE int atm_cas_acq_u32_ (nsync_atomic_uint32_ *p, uint32_t o, uint32_t n) {
	int res = (atomic_cas_32 (NSYNC_ATOMIC_UINT32_PTR_ (p), o, n) == o);
	membar_enter ();
	return (res);
}

#define ATM_CAS(p,o,n)        (atm_cas_nomb_u32_ ((p),(o),(n)))
#define ATM_CAS_ACQ(p,o,n)    (atm_cas_acq_u32_ ((p),(o),(n)))
#define ATM_CAS_REL(p,o,n)    (membar_exit (), ATM_CAS ((p), (o), (n)))
#define ATM_CAS_RELACQ(p,o,n) (membar_exit (), atm_cas_acq_u32_ ((p),(o),(n)))

/*----*/

static INLINE uint32_t atm_load_acq_u32_ (nsync_atomic_uint32_ *p) {
	uint32_t result = NSYNC_ATOMIC_UINT32_LOAD_ ((volatile nsync_atomic_uint32_ *)p);
	ATM_LD_ACQ_ ();
	return (result);
}

#define ATM_LOAD(p) (NSYNC_ATOMIC_UINT32_LOAD_ ((volatile nsync_atomic_uint32_ *)(p)))
#define ATM_LOAD_ACQ(p) (atm_load_acq_u32_ (p))

/*----*/

#define ATM_STORE_X_(p,v,b,a) do { \
		b; \
		NSYNC_ATOMIC_UINT32_STORE_ ((volatile nsync_atomic_uint32_ *)(p), (uint32_t) (v)); \
		a; \
	} while (0)

#define ATM_STORE(p,v)     ATM_STORE_X_ ((p), (v), ;             , ;       )
#define ATM_STORE_REL(p,v) ATM_STORE_X_ ((p), (v), ATM_ST_REL_ (), ;       )

NSYNC_CPP_END_

#endif /*NSYNC_PLATFORM_NETBSD_ATOMIC_H_*/
