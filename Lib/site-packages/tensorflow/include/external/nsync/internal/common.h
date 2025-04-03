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

#ifndef NSYNC_INTERNAL_COMMON_H_
#define NSYNC_INTERNAL_COMMON_H_

#include "nsync_cpp.h"
#include "platform.h"
#include "nsync_atomic.h"
#include "sem.h"
#include "nsync_waiter.h"
#include "dll.h"
#include "nsync_mu.h"
#include "nsync_note.h"

/* Annotations for race detectors. */
#if defined(__has_feature) && !defined(__SANITIZE_THREAD__)
#if __has_feature(thread_sanitizer)  /* used by clang */
#define __SANITIZE_THREAD__ 1 /* GCC uses this; fake it in clang */
#endif
#endif
#if defined(__SANITIZE_THREAD__)
NSYNC_C_START_
void AnnotateIgnoreWritesBegin (const char* file, int line);
void AnnotateIgnoreWritesEnd (const char* file, int line);
void AnnotateIgnoreReadsBegin (const char* file, int line);
void AnnotateIgnoreReadsEnd (const char* file, int line);
void AnnotateRWLockCreate (const char *file, int line, void *mu);
void AnnotateRWLockAcquired (const char *file, int line, void *mu, long write);
void AnnotateRWLockReleased (const char *file, int line, void *mu, long write);
NSYNC_C_END_
#define RWLOCK_CREATE(mu_) AnnotateRWLockCreate (__FILE__, __LINE__, (mu_))
#define RWLOCK_TRYACQUIRE(cond_, mu_, wrt_) do { if (cond_) { AnnotateRWLockAcquired (__FILE__, __LINE__, (mu_), (wrt_)); } } while (0)
#define RWLOCK_RELEASE(mu_, wrt_) do { AnnotateRWLockReleased (__FILE__, __LINE__, (mu_), (wrt_)); } while (0)
#define IGNORE_RACES_START() \
	do { \
		AnnotateIgnoreReadsBegin (__FILE__, __LINE__); \
		AnnotateIgnoreWritesBegin (__FILE__, __LINE__); \
	} while (0)
#define IGNORE_RACES_END() \
	do { \
		AnnotateIgnoreWritesEnd (__FILE__, __LINE__); \
		AnnotateIgnoreReadsEnd (__FILE__, __LINE__); \
	} while (0)
#else
#define RWLOCK_CREATE(mu_)
#define RWLOCK_TRYACQUIRE(cond_, mu_, wrt_)
#define RWLOCK_RELEASE(mu_, wrt_)
#define IGNORE_RACES_START()
#define IGNORE_RACES_END()
#endif

#ifndef NSYNC_DEBUG
#define NSYNC_DEBUG 0
#endif

NSYNC_CPP_START_

/* Yield the CPU. Platform specific. */
void nsync_yield_ (void);

/* Retrieve the per-thread cache of the waiter object.  Platform specific. */
void *nsync_per_thread_waiter_ (void (*dest) (void *));

/* Set the per-thread cache of the waiter object.  Platform specific. */
void nsync_set_per_thread_waiter_ (void *v, void (*dest) (void *));

/* Used in spinloops to delay resumption of the loop.
   Usage:
       unsigned attempts = 0;
       while (try_something) {
	  attempts = nsync_spin_delay_ (attempts);
       } */
unsigned nsync_spin_delay_ (unsigned attempts);

/* Spin until (*w & test) == 0, then atomically perform *w = ((*w | set) &
   ~clear), perform an acquire barrier, and return the previous value of *w.
   */
uint32_t nsync_spin_test_and_set_ (nsync_atomic_uint32_ *w, uint32_t test,
				   uint32_t set, uint32_t clear);

/* Abort after printing the nul-temrinated string s[]. */
void nsync_panic_ (const char *s);

/* ---------- */

#define MIN_(a_,b_) ((a_) < (b_)? (a_) : (b_))
#define MAX_(a_,b_) ((a_) > (b_)? (a_) : (b_))

/* ---------- */

/* Fields in nsync_mu.word.

   - At least one of the MU_WLOCK or MU_RLOCK_FIELD fields must be zero.
   - MU_WLOCK indicates that a write lock is held.
   - MU_RLOCK_FIELD is a count of readers with read locks.

   - MU_SPINLOCK represents a spinlock that must be held when manipulating the
     waiter queue.

   - MU_DESIG_WAKER indicates that a former waiter has been woken, but has
     neither acquired the lock nor gone back to sleep.  Legal to fail to set it;
     illegal to set it when no such waiter exists.

   - MU_WAITING indicates whether the waiter queue is non-empty.
     The following bits should be zero if MU_WAITING is zero.
   - MU_CONDITION indicates that some waiter may have an associated condition
     (from nsync_mu_wait, etc.).  Legal to set it with no such waiter exists,
     but illegal to fail to set it with such a waiter.
   - MU_WRITER_WAITING indicates that a reader that has not yet blocked
     at least once should not acquire in order not to starve waiting writers.
     It set when a writer blocks or a reader is woken with a writer waiting.
     It is reset when a writer acquires, but set again when that writer
     releases if it wakes readers and there is a waiting writer.
   - MU_LONG_WAIT indicates that a waiter has been woken many times but
     repeatedly failed to acquire when competing for the lock.  This is used
     only to prevent long-term starvation by writers.  The thread that sets it
     clears it when if acquires.
   - MU_ALL_FALSE indicates that a complete scan of the waiter list found no
     waiters with true conditions, and the lock has not been acquired by a
     writer since then.  This allows a reader lock to be released without
     testing conditions again.  It is legal to fail to set this, but illegal
     to set it inappropriately.
 */
#define MU_WLOCK ((uint32_t) (1 << 0)) /* writer lock is held. */
#define MU_SPINLOCK ((uint32_t) (1 << 1)) /* spinlock is held (protects waiters). */
#define MU_WAITING ((uint32_t) (1 << 2)) /* waiter list is non-empty. */
#define MU_DESIG_WAKER ((uint32_t) (1 << 3)) /* a former waiter awoke, and hasn't yet acquired or slept anew */
#define MU_CONDITION ((uint32_t) (1 << 4)) /* the wait list contains some conditional waiters. */
#define MU_WRITER_WAITING ((uint32_t) (1 << 5)) /* there is a writer waiting */
#define MU_LONG_WAIT ((uint32_t) (1 << 6)) /* the waiter at the head of the queue has been waiting a long time */
#define MU_ALL_FALSE ((uint32_t) (1 << 7)) /* all waiter conditions are false */
#define MU_RLOCK ((uint32_t) (1 << 8)) /* low-order bit of reader count, which uses rest of word */

/* The constants below are derived from those above. */
#define MU_RLOCK_FIELD (~(uint32_t) (MU_RLOCK - 1)) /* mask of reader count field */

#define MU_ANY_LOCK (MU_WLOCK | MU_RLOCK_FIELD) /* mask for any lock held */

#define MU_WZERO_TO_ACQUIRE (MU_ANY_LOCK | MU_LONG_WAIT) /* bits to be zero to acquire write lock */
#define MU_WADD_TO_ACQUIRE (MU_WLOCK)         /* add to acquire a write lock */
#define MU_WHELD_IF_NON_ZERO (MU_WLOCK)       /* if any of these bits are set, write lock is held */
#define MU_WSET_WHEN_WAITING (MU_WAITING | MU_WRITER_WAITING) /* a writer is waiting */
#define MU_WCLEAR_ON_ACQUIRE (MU_WRITER_WAITING)  /* clear MU_WRITER_WAITING when a writer acquires */
#define MU_WCLEAR_ON_UNCONTENDED_RELEASE (MU_ALL_FALSE) /* clear if a writer releases w/o waking */

/* bits to be zero to acquire read lock */
#define MU_RZERO_TO_ACQUIRE (MU_WLOCK | MU_WRITER_WAITING | MU_LONG_WAIT)
#define MU_RADD_TO_ACQUIRE (MU_RLOCK)         /* add to acquire a read lock */
#define MU_RHELD_IF_NON_ZERO (MU_RLOCK_FIELD) /* if any of these bits are set, read lock is held */
#define MU_RSET_WHEN_WAITING (MU_WAITING)     /* indicate that some thread is waiting */
#define MU_RCLEAR_ON_ACQUIRE ((uint32_t) 0)              /* nothing to clear when a read acquires */
#define MU_RCLEAR_ON_UNCONTENDED_RELEASE ((uint32_t) 0)  /* nothing to clear when a read releases */


/* A lock_type holds the values needed to manipulate a mu in some mode (read or
   write).  This allows some of the code to be generic, and parameterized by
   the lock type. */
typedef struct lock_type_s {
	uint32_t zero_to_acquire; /* bits that must be zero to acquire */
	uint32_t add_to_acquire; /* constant to add to acquire */
	uint32_t held_if_non_zero; /* if any of these bits are set, the lock is held */
	uint32_t set_when_waiting; /* set when thread waits */
	uint32_t clear_on_acquire; /* clear when thread acquires */
	uint32_t clear_on_uncontended_release; /* clear when thread releases without waking */
} lock_type;


/* writer_type points to a lock_type that describes how to manipulate a mu for a writer. */
extern lock_type *nsync_writer_type_;

/* reader_type points to a lock_type that describes how to manipulate a mu for a reader. */
extern lock_type *nsync_reader_type_;

/* ---------- */

/* Bits in nsync_cv.word */

#define CV_SPINLOCK ((uint32_t) (1 << 0)) /* protects waiters */
#define CV_NON_EMPTY ((uint32_t) (1 << 1)) /* waiters list is non-empty */

/* ---------- */

/* Hold a pair of  condition function and its argument. */
struct wait_condition_s {
	int (*f) (const void *v);
	const void *v;
	int (*eq) (const void *a, const void *b);
};

/* Return whether wait conditions *a_ and *b_ are equal and non-null. */
#define WAIT_CONDITION_EQ(a_, b_)  ((a_)->f != NULL && (a_)->f == (b_)->f && \
                                    ((a_)->v == (b_)->v || \
				     ((a_)->eq != NULL && (*(a_)->eq) ((a_)->v, (b_)->v))))

/* If a waiter has waited this many times, it may set the MU_LONG_WAIT bit. */
#define LONG_WAIT_THRESHOLD 30

/* ---------- */

#define NOTIFIED_TIME(n_) (ATM_LOAD_ACQ (&(n_)->notified) != 0? nsync_time_zero : \
			   (n_)->expiry_time_valid? (n_)->expiry_time : nsync_time_no_deadline)

/* A waiter represents a single waiter on a cv or a mu.

   To wait:
   Allocate a waiter struct *w with new_waiter(), set w.waiting=1, and
   w.cv_mu=nil or to the associated mu if waiting on a condition variable, then
   queue w.nsync_dll on some queue, and then wait using:
      while (ATM_LOAD_ACQ (&w.waiting) != 0) { nsync_mu_semaphore_p (&w.sem); }
   Return *w to the freepool by calling free_waiter (w).

   To wakeup:
   Remove *w from the relevant queue then:
    ATM_STORE_REL (&w.waiting, 0);
    nsync_mu_semaphore_v (&w.sem); */
typedef struct {
	uint32_t tag;              /* debug DLL_NSYNC_WAITER, DLL_WAITER, DLL_WAITER_SAMECOND */
	nsync_semaphore sem;       /* Thread waits on this semaphore. */
	struct nsync_waiter_s nw;  /* An embedded nsync_waiter_s. */
	struct nsync_mu_s_ *cv_mu;  /* pointer to nsync_mu associated with a cv wait */
	lock_type *l_type;         /* Lock type of the mu, or nil if not associated with a mu. */
	nsync_atomic_uint32_ remove_count;   /* count of removals from queue */
	struct wait_condition_s cond; /* A condition on which to acquire a mu. */
	nsync_dll_element_ same_condition;   /* Links neighbours in nw.q with same non-nil condition. */
	int flags;                    /* see WAITER_* bits below */
} waiter;
static const uint32_t WAITER_TAG = 0x0590239f;
static const uint32_t NSYNC_WAITER_TAG = 0x726d2ba9;

#define WAITER_RESERVED 0x1  /* waiter reserved by a thread, even when not in use */
#define WAITER_IN_USE   0x2  /* waiter in use by a thread */

#define CONTAINER(t_,f_,p_)  ((t_ *) (((char *) (p_)) - offsetof (t_, f_)))
#define ASSERT(x) do { if (!(x)) { *(volatile int *)0 = 0; } } while (0)
	
/* Return a pointer to the nsync_waiter_s containing nsync_dll_element_ *e. */
#define DLL_NSYNC_WAITER(e) (NSYNC_DEBUG? nsync_dll_nsync_waiter_ (e) : \
	((struct nsync_waiter_s *)((e)->container)))
struct nsync_waiter_s *nsync_dll_nsync_waiter_ (nsync_dll_element_ *e);

/* Return a pointer to the waiter struct that *e is embedded in, where *e is an nw.q field. */
#define DLL_WAITER(e) (NSYNC_DEBUG? nsync_dll_waiter_ (e) : \
	CONTAINER (waiter, nw, DLL_NSYNC_WAITER(e)))
waiter *nsync_dll_waiter_ (nsync_dll_element_ *e);

/* Return a pointer to the waiter struct that *e is embedded in, where *e is a
   same_condition field.  */
#define DLL_WAITER_SAMECOND(e) (NSYNC_DEBUG? nsync_dll_waiter_samecond_ (e) : \
	((waiter *) ((e)->container)))
waiter *nsync_dll_waiter_samecond_ (nsync_dll_element_ *e);

/* Return a pointer to an unused waiter struct.
   Ensures that the enclosed timer is stopped and its channel drained. */
waiter *nsync_waiter_new_ (void);

/* Return an unused waiter struct *w to the free pool. */
void nsync_waiter_free_ (waiter *w);

/* ---------- */

/* The internals of an nync_note.  See internal/note.c for details of locking
   discipline.  */
struct nsync_note_s_ {
        nsync_dll_element_ parent_child_link; /* parent's children, under parent->note_mu  */
        int expiry_time_valid;      /* whether expiry_time is valid; r/o after init */
        nsync_time expiry_time;     /* expiry time, if expiry_time_valid != 0; r/o after init */
        nsync_mu note_mu;          /* protects fields below except "notified" */
        nsync_cv no_children_cv;    /* signalled when children becomes empty */
        uint32_t disconnecting;     /* non-zero => node is being disconnected */
        nsync_atomic_uint32_ notified;   /* non-zero if the note has been notified */
        struct nsync_note_s_ *parent;     /* points to parent, if any */
        nsync_dll_element_ *children; /* list of children */
        nsync_dll_element_ *waiters;  /* list of waiters */
};

/* ---------- */

void nsync_mu_lock_slow_ (nsync_mu *mu, waiter *w, uint32_t clear, lock_type *l_type);
void nsync_mu_unlock_slow_ (nsync_mu *mu, lock_type *l_type);
nsync_dll_list_ nsync_remove_from_mu_queue_ (nsync_dll_list_ mu_queue, nsync_dll_element_ *e);
void nsync_maybe_merge_conditions_ (nsync_dll_element_ *p, nsync_dll_element_ *n);
nsync_time nsync_note_notified_deadline_ (nsync_note n);
int nsync_sem_wait_with_cancel_ (waiter *w, nsync_time abs_deadline,
				 nsync_note cancel_note);
NSYNC_CPP_END_

#endif /*NSYNC_INTERNAL_COMMON_H_*/
